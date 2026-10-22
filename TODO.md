# TODO

## Fatto

- [x] **Riorganizzare la struttura degli header e i namespace**
  - Rimosso il namespace e la directory `builtin`; le implementazioni sono organizzate per area in `activation`, `loss`, `net`, `rule` e `backprop`.
  - Tutto il backpropagation è sotto `rules/backprop/`; il backend OpenMP è un template sul tipo di optimizer, che possiede e inizializza direttamente.
  - La rule delega il lavoro al backend tramite i concept `BackendType`; gradienti e aggiornamento dei pesi sono gestiti nell'operazione di backprop del backend.

- [x] **Disaccoppiare il backpropagation dal tipo concreto del neurone**
  - `Neuron` non è più templata sull'attivazione: `BackpropNeuron`/`PerceptronNeuron` sono stati rimossi e ogni neurone possiede un'istanza polimorfica di `Activation`.
  - La logica di apprendimento è estratta nelle learning rule (`rule::Backprop`, `rule::Perceptron`), che operano su `NNet::View`/`Neuron`.
  - Reti con attivazioni diverse per neurone sono già possibili.

- [x] **Separare il calcolo dei gradienti dall'aggiornamento dei pesi**
  - Il backend calcola e retro-propaga i delta, accumula i gradienti e affida l'aggiornamento all'optimizer.
  - Il learning rate è configurato dall'optimizer Momentum; la rule Perceptron riceve il proprio learning rate.
  - Gli optimizer mantengono il proprio stato per peso; mini-batch e gradient accumulation sono stati implementati.

- [x] **Rendere espliciti input, output e sorgenti della rete**
  - I neuroni sono ora etichettati con `NeuronType` (`input`, `hidden`, `output`, `bias`), assegnato esplicitamente alla creazione con `new_neurons(...)`.
  - `NNet::train()` e `NNet::infer()` selezionano i neuroni per tipo e validano che il conteggio corrisponda a `input_size`/`output_size`.
  - Non c'è più alcuna deduzione basata su connessioni entranti/uscenti.

- [x] **Gestire il bias come parte della rete, non come dato di training**
  - I neuroni di bias vengono creati con `new_neurons(..., NeuronType::bias, ...)`, collegati ai receiver e inizializzati con segnale costante 1.0 durante la propagazione.
  - Il bias non è parte dei sample di training: la topologia della rete non influenza la forma dei dati.

- [x] **Unificare gli overload di `connect`**
  - `Topology::connect(tx, rx)` crea il prodotto cartesiano tra due range di indici, con un'unica implementazione basata sul concept `IndexRange`.
  - `connect()` e `zip_connect()` accettano range non const, incluse viste iterabili solo da non-const.
  - `zip_connect()` collega gli elementi a coppie in lockstep e rileva lunghezze diverse durante l'iterazione, senza richiedere `size()`.

- [x] **Compilare la topologia per il forward/backward pass**
  - Il costruttore di `NNet` ordina le connessioni per receiver/transmitter e costruisce partizioni contigue; quello di `DAGNet` rifiuta i cicli e ordina le partizioni topologicamente.
  - Partizioni dense compatibili vengono riconosciute come `DAGNet::Layer` e percorse come prodotti matrice-vettore con pesi e segnali contigui.

- [x] **Separare i segnali dalla struttura del neurone**
  - Segnali e weighted sum vivono in `NNet::State` in vettori allineati agli indici dei neuroni; lo stato è separato dalla topologia e può essere istanziato per ogni thread di training.
  - `Neuron` conserva tipo e attivazione, non i segnali o i weighted sum.

- [x] **Introdurre mini-batch e gradient accumulation**
  - `rule::Backprop` divide i sample in mini-batch; il backend OpenMP elabora i sample in parallelo, accumulando i gradienti in buffer per-thread.
  - I gradienti dei thread vengono sommati e l'optimizer aggiorna i pesi alla fine di ogni mini-batch; lo stato per-thread (`NNet::State`: segnali e weighted sum) è separato dalla topologia.
  - L'optimizer possiede gli iperparametri dell'algoritmo, incluso il learning rate per Momentum.
  
- [x] **Rendere operativo `zip_connect()`**
  - Collega gli elementi corrispondenti in lockstep; range di lunghezza diversa generano un'eccezione dopo aver processato il prefisso comune.
    
- [x] **Rilevare connessioni duplicate prima di usare la rete**
  - `NNet::init()`, privata e chiamata dal costruttore, controlla tutte le connessioni, incluse quelle inserite direttamente in `Topology`, usando una hash map temporanea.
  - Anche gli estremi delle connessioni vengono validati nel costruttore, non negli helper; non viene mantenuto un indice dei duplicati persistente nella rete.
    
- [x] **Implementare un backend Metal oltre a OpenMP**
  - Backend Objective-C++ per `DenseForwardNet`, con prodotti matriciali MPS e kernel di attivazione, derivate e ottimizzazione.
  - Supportati FP32, Identity/Sigmoid/ReLU, MSE e optimizer Momentum/Rprop; bias incluso nelle colonne delle matrici, senza kernel separati.
     
- [x] **Use literals**
  - I literals consentono di poter cambiare in modo immediato i tipi di dati (e.g. real_t da 64bit a 32bit per es) utile per fare test e prove in modo veloce
    
- [x] **Separare la costruzione della topologia dall'esecuzione della rete**
  - `Topology` contiene neuroni e connessioni pubblici e gli helper `new_neurons()`, `connect()` e `zip_connect()`.
  - `NNet` acquisisce i vettori e prepara i dati comuni nel costruttore; `DAGNet` completa validazione e preparazione DAG nel proprio costruttore, senza hook virtuale `init()`.
  - `DenseForwardNet` mantiene il costruttore variadico pubblico e prepara la topologia con una funzione statica privata, creando solo i bias richiesti prima dei riceventi.
  - La View espone dati strutturali in sola lettura e pesi modificabili. Resta protetto l'accesso mutabile alle partizioni per le implementazioni derivate.

- [x] **Semplificare il rilevamento dei layer densi in DAGNet**
  - Raggruppamento diretto per intervallo sorgente, senza UnionFind; i gruppi mantengono l'ordine di prima apparizione nelle partizioni topologiche.
  - Conservati i controlli di contiguita e la mappatura dalle partizioni ai layer.

- [x] **Riconoscere i layer densi con un solo ricevente in DAGNet**
  - Rimosso il filtro sui gruppi con meno di due partizioni; le partizioni con sorgenti non contigue sono gia escluse durante il raggruppamento.
  - I layer di output con un solo neurone sono inclusi nei descrittori esposti anche al backend Metal.

- [x] **Esporre la mappatura tra partizioni e layer nella View**
  - `layers()` restituisce uno span const ordinato topologicamente; `partition_layer_indices()` associa ogni partizione a un layer o a `Layer::no_layer`.
  - Rimosso `is_dense()`; OpenMP usa la mappatura senza cursori sincronizzati, mentre Metal percorre direttamente i layer e usa gli indici di input e bias esposti da `NNet::View`.

## Priorita alta

- [ ] **Aggiungere optimizer adattivi**
  - Implementare Adam e valutare RMSProp; Rprop è già disponibile come optimizer OpenMP.
  - I nuovi optimizer devono mantenere stato per peso e funzionare con i mini-batch.

- [ ] **Unificare le formule di attivazioni e loss tra CPU e backend GPU**
  - Metal implementa separatamente Identity, Sigmoid, ReLU e MSE: formule, derivate e gestione dei parametri sono duplicate rispetto al codice CPU.
  - Valutare una sorgente comune delle formule, con adattamenti per le funzioni matematiche e i vincoli di Metal ed eventuali altri backend, evitando implementazioni indipendenti che possano divergere.
  - Definire come estendere il supporto ad attivazioni e loss personalizzate; verificare la coerenza di valori e derivate tra backend con tolleranze adeguate alla precisione numerica.

## Discussioni aperte

- [ ] **Separare l'update della rete in backend di esecuzione riutilizzabili**
  - La rete possiede modello, pesi e metadati; la View espone i dati ai backend, che eseguono le operazioni senza richiamare `net.update()`.
  - La View serve a separare l'accesso ai dati dall'API operativa della rete: deve esporre solo dati e metadati, senza `infer()`, `train()` o `update()`.
  - Solo l'utente utilizzatore della rete invoca le sue operazioni pubbliche, come `infer()` e `train()`. Le implementazioni interne di learning rule e backend devono accedere alla rete esclusivamente tramite View, non tramite un riferimento alla rete completa; adeguare anche `init()`, `to_device()` e `to_host()`.
  - Estrarre l'update CPU attualmente in `DAGNet` e quello GPU attualmente nel backend Metal di backprop in implementazioni indipendenti dalle learning rule e dagli optimizer.
  - Rendere `update()` e `infer()` templati sul backend, passato per riferimento per riutilizzarne le risorse; preservare la View specifica della rete e i vincoli sui modelli supportati.
  - Il backend di backprop integra internamente il backend di update dello stesso dispositivo; conservare l'API pubblica `backprop::omp::Backend<Optimizer>` e l'equivalente Metal.
  - Su Metal condividere buffer e command buffer tra update e backward, evitando trasferimenti CPU/GPU o allocazioni duplicate. Non imporre lo stato host `NNet::State` a ogni implementazione.
  - Usare il nome generico `update`: per una DAG e una propagazione topologica, per altre reti puo essere un passo temporale, un aggiornamento sincrono/asincrono o un campionamento.
  - Non obbligare ogni learning rule a possedere un backend di update: alcune regole possono determinare i pesi direttamente dai dati.
  - Verificare il design prima su update DAG, backprop, Perceptron e inferenza; valutare in seguito reti ricorrenti, Hopfield e Boltzmann senza anticipare astrazioni non necessarie.

## Priorita bassa

- [ ] **Migliorare la configurazione dell'addestramento**
  - Rendere configurabili soglia di errore e numero massimo di epoche; batch size, loss e backend/optimizer sono configurabili in `rule::Backprop`.
  - Valutare metriche e callback separati dal logging diretto su stdout.

## Performance

- [ ] **Ottimizzare i kernel dei dense block sulla base del profiling**
  - I tempi storici (rete 784→128→10) non sono una baseline della configurazione MNIST corrente (784→512→512→512→10); usare benchmark riproducibili prima di confrontare modifiche.
  - Le direttive SIMD e l'unrolling manuale vanno mantenuti solo dove il benchmark dimostra un vantaggio.
  - Valutare layout, blocking e riduzione del traffico read-modify-write nel passo di update prima di introdurre rappresentazioni duplicate dei pesi.

- [ ] **Ridurre il dispatch di attivazione nei batch densi**
  - `Neuron` conserva un puntatore polimorfico a `Activation`; il dispatch avviene una volta per neurone attivato.
  - Raggruppare neuroni/layer per attivazione può rendere possibile applicare la funzione in batch su dati contigui.

## Stress test / Benchmark

- [x] **Aggiungere un esempio MNIST con una rete più grande**
  - `examples/backprop/src/backprop_mnist.cpp` allena un MLP 784→512→512→512→10 usando fino a 60k sample di training e 10k di test.

- [ ] **Rendere riproducibili i benchmark di performance**
  - Registrare configurazione CPU, compilatore, flag Release e seed del generatore casuale.
  - Misurare più epoch e riportare mediana o media, separando setup della topologia, forward, delta e weight update.

## Report di profiling storico (backprop_mnist, 2026-08-24)

Questo report descrive l'implementazione precedente, basata su adjacency list e propagazione ricorsiva. I risultati non sono direttamente confrontabili con la versione corrente a partition/dense block né con l'esempio MNIST attuale; resta come traccia della procedura Callgrind.

Contesto: si sospettava che `BackpropRule::learn()` (nell'implementazione storica di `backprop_rule.h`) fosse il collo di
bottiglia del training MNIST (~6 minuti per l'intero dataset, 60k campioni, rete 784→100→10).
Prima ipotesi testata: la ricorsione in `learn()`/`broadcast()` come causa di overhead — **non
confermata**: convertire `backprop_error` da ricorsivo a iterativo (worklist esplicito) non ha
prodotto variazioni misurabili di performance (compilando con `-O3` il compilatore ottimizza
già la ricorsione in modo simile a un ciclo; la profondità di ricorsione per questa rete è
comunque minima, pari al numero di layer).

### Procedura di profiling usata

Il vincolo principale è stato isolare la fase di training (`NNet::train`/`learn`/`broadcast`)
dalla fase di setup della rete (`NNet::connect`), che per una rete densa 784×100 = 78'400
connessioni ha un costo **O(n²)** dovuto al controllo di duplicati in `connect()`
(`std::ranges::any_of` su tutte le connessioni già inserite). Sotto callgrind (instrumentazione
completa, ~50-100x più lento del nativo) questo setup da solo satura qualunque timeout
ragionevole, mascherando il costo reale del training.

Passi seguiti:
1. Ridotto temporaneamente dataset/rete per l'esecuzione sotto valgrind (`nb_training_samples`,
   `nb_hidden`), per portare il tempo di esecuzione in un intervallo pratico (l'algoritmo O(n²)
   di `connect()` resta comunque presente, solo più piccolo).
2. Aggiunto `#include <valgrind/callgrind.h>` e circondata la chiamata a `classifier.train(...)`
   con `CALLGRIND_START_INSTRUMENTATION;` / `CALLGRIND_STOP_INSTRUMENTATION;` in
  `backprop_mnist.cpp` (modifica temporanea, poi rimossa).
3. Eseguito con `valgrind --tool=callgrind --instr-atstart=no ...`: la raccolta dati resta
   disattivata (quindi quasi a costo zero) durante lettura CSV e `connect()`, e si attiva solo
   dentro `train()`.
4. Analizzato l'output con `callgrind_annotate` (per funzione) e per riga (`--tree=both` per
   il call-tree). Compilazione con `-g` necessaria per avere nomi di funzione/riga leggibili
   invece di indirizzi grezzi.
5. Profiling di memoria complementare con `valgrind --tool=massif` (nessuna anomalia rilevante:
   uso heap contenuto, coerente con le dimensioni di rete/dataset usate per il test).

È stato creato uno script (`scripts/profile_callgrind.sh`) che automatizza i passi 3-4 per
riusi futuri: vedi intestazione dello script per l'uso.

### Risultati (rete 784→20→10, 1000 campioni di training, ~50 miliardi di istruzioni Ir
raccolte solo dentro `train()`)

| Componente | % Ir (istruzioni) |
|---|---|
| **`BackpropRule::learn()` totale** (retropropagazione errori + aggiornamento pesi) | **~38-43%** |
| — aggiornamento pesi: `conn.weight += lr*delta*signal + momentum*dweight` | ~10.5% |
| — retropropagazione: `_deltas[conn.itx] += _deltas[ineuron]*conn.weight` | ~7.0% |
| — conteggio delta ricevuti: `++_number_rx_deltas[...] == out_connections.size()` | ~5.8% |
| — overhead di iterazione del ciclo di aggiornamento pesi | ~4.7% |
| **`NNet::broadcast()` (forward pass)** | **~12%** |
| — loop `_out_connections[itx]` + `receive_signal` | ~6.5% |
| **Indirection extra tramite `View` per accedere a `_connections`/`_neurons`** | ~5% |
| Overhead sparso di iterator/vector (STL, ranges) | ~30% |

### Conclusioni

1. `learn()` è effettivamente il costo dominante per-sample, ma non per via della ricorsione:
   è il volume di lavoro (due passate su tutte le connessioni, ciascuna con accessi indicizzati
   a `_deltas`, `_dweights`, `_connections`, `_neurons`) a pesare.
2. Il forward pass (`broadcast`) non è trascurabile (~12%), circa un terzo del costo di `learn()`.
3. Una quota consistente (~30%) è overhead "strutturale": ogni accesso ad arco/neurone passa
   per `View` → riferimento a `NNet` → `vector<...>`, con doppia indirezione ripetuta invece
   di un accesso diretto.
4. La rete di test è piccola abbastanza da restare in cache L2/L3: **non è un problema di
   cache-miss**, è puro volume di istruzioni dovuto a una rappresentazione a grafo generico
   (adjacency list + `variant` per l'attivazione) invece che a operazioni matriciali.
5. **La vera leva di performance sarebbe strutturale**: essendo di fatto un MLP fully-connected
   a due layer, rappresentare pesi/attivazioni come buffer contigui per layer (matrice densa)
   permetterebbe l'auto-vettorizzazione SIMD dei loop, cosa che l'attuale attraversamento a
   grafo con indirection tramite `View` impedisce strutturalmente. Cambiamento architetturale
   più ampio rispetto a una micro-ottimizzazione locale di `learn()`.
6. Nota collaterale (non nel path di training, quindi fuori scope per questo report ma già
   tracciata sopra in "Performance"): il costo O(n²) di `connect()` per reti dense è reale e
   ha reso necessario il lavoro di isolamento descritto sopra pur di poter profilare `train()`.

## Nota

La classe `BiasNeuron` non esiste nella versione corrente: il bias e un `Neuron`
di tipo `NeuronType::bias`, creato tramite `Topology::new_neurons()` e inizializzato
con segnale 1 per l'attivazione Identity. Non fa parte dei dati di training.
