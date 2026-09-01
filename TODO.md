# TODO

## Fatto

- [x] **Disaccoppiare il backpropagation dal tipo concreto del neurone**
  - `Neuron` non è più templata sull'attivazione: `BackpropNeuron`/`PerceptronNeuron` sono stati rimossi e ogni neurone possiede un'istanza polimorfica di `Activation`.
  - La logica di apprendimento è estratta nelle learning rule (`BackpropRule`, `PerceptronRule`), che operano su `NNet::View`/`Neuron`.
  - Reti con attivazioni diverse per neurone sono già possibili.

- [x] **Separare il calcolo dei gradienti dall'aggiornamento dei pesi**
  - In `BackpropRule::learn()` i delta vengono calcolati e retro-propagati in un primo passaggio, l'aggiornamento dei pesi avviene in un secondo ciclo separato.
  - Il learning rate è ora un parametro del costruttore di `BackpropRule`/`PerceptronRule`, non più hardcoded.
  - `BackpropRule` mantiene anche lo stato del momentum. Restano da implementare batch/gradient accumulation e optimizer adattivi.

- [x] **Rendere espliciti input, output e sorgenti della rete**
  - I neuroni sono ora etichettati con `NeuronType` (`input`, `hidden`, `output`, `bias`), assegnato esplicitamente alla creazione con `new_neurons(...)`.
  - `NNet::train()` e `NNet::infer()` selezionano i neuroni per tipo e validano che il conteggio corrisponda a `input_size`/`output_size`.
  - Non c'è più alcuna deduzione basata su connessioni entranti/uscenti.

- [x] **Gestire il bias come parte della rete, non come dato di training**
  - `NNet::add_bias(neurons)` crea un neurone di bias per ciascun neurone del range passato, lo collega e lo attiva con segnale costante 1.0.
  - Il bias non è più parte di `TrainingSample.inputs`: la topologia della rete non influenza più la forma dei dati di training.

- [x] **Unificare gli overload di `connect`**
  - `NNet::connect(tx, rx)` accetta ora, per ciascun lato, sia un singolo `Neuron` sia un `NeuronRange`, tramite un'unica funzione template invece di quattro overload distinti.
  - Internamente entrambi gli argomenti vengono uniformati a una view di indici per riutilizzare la stessa logica di controllo/creazione delle connessioni.

- [x] **Compilare la topologia per il forward/backward pass**
  - Prima del training, `prepare()` ordina le connessioni per receiver, costruisce partition contigue e le ordina topologicamente; reti cicliche vengono rifiutate.
  - Partition dense compatibili vengono riconosciute come `DenseBlock` e percorse come prodotti matrice-vettore con pesi e segnali contigui.

- [x] **Separare i segnali dalla struttura del neurone**
  - I segnali vivono in `NNet::_signals`, un `std::vector<real_t>` allineato agli indici dei neuroni; `Neuron` conserva soltanto metadati, attivazione e weighted sum.

## Priorita alta

- [ ] **Introdurre mini-batch e gradient accumulation**
  - Accumulare gradienti per un batch di sample e aggiornare i pesi una volta per batch.
  - Separare completamente lo stato per-sample (segnali, weighted sum e delta) dalla topologia, in modo da poter rappresentare buffer `batch × neuroni` e sfruttare kernel densi.

- [ ] **Aggiungere optimizer adattivi**
  - Implementare Adam come primo optimizer adattivo; valutare RMSProp e RPROP in seguito.
  - Gli optimizer devono mantenere stato per peso e funzionare sia con SGD sia con mini-batch.

## Priorita bassa

- [ ] **Introdurre un'astrazione per layer o modelli di rete**
  - Oggi l'utente deve creare e collegare manualmente ogni gruppo di neuroni.
  - Aggiungere helper per costruire layer densi e collegamenti tra layer ridurrebbe il codice ripetitivo senza nascondere il grafo quando serve controllo fine.

- [ ] **Migliorare la configurazione dell'addestramento**
  - Rendere configurabili soglia di errore, numero massimo di epoche, learning rate e strategia di aggiornamento.
  - Valutare metriche e callback separati dal logging diretto su stdout.

## Performance

- [ ] **Rendere O(1) il controllo di connessione duplicata in `connect()`**
  - Attualmente `std::ranges::any_of` scansiona tutte le connessioni esistenti per ogni nuova coppia (tx, rx): O(n²) nella costruzione di layer densi.
  - Sostituire con una hash set su `(itx, irx)`.

- [ ] **Ottimizzare i kernel dei dense block sulla base del profiling**
  - La baseline Release per MNIST (60k sample, rete 784→128→10) è circa 6s per epoca: forward ~1.1s, backprop delta ~1.9s, update pesi ~3.0s.
  - Il loop del delta dense è più veloce senza `#pragma omp simd`; le direttive SIMD e l'unrolling manuale vanno mantenuti solo dove il benchmark dimostra un vantaggio.
  - Valutare layout, blocking e riduzione del traffico read-modify-write nel passo di update prima di introdurre rappresentazioni duplicate dei pesi.

- [ ] **Ridurre il dispatch di attivazione nei batch densi**
  - `Neuron` conserva un puntatore polimorfico a `Activation`; il dispatch avviene una volta per neurone attivato.
  - Raggruppare neuroni/layer per attivazione può rendere possibile applicare la funzione in batch su dati contigui.

## Stress test / Benchmark

- [x] **Aggiungere un esempio MNIST con una rete più grande**
  - `examples/backprop/src/backprop-mnist.cpp` allena un MLP 784→128→10 sui 60k sample MNIST e misura il tempo per epoca.

- [ ] **Rendere riproducibili i benchmark di performance**
  - Registrare configurazione CPU, compilatore, flag Release e seed del generatore casuale.
  - Misurare più epoch e riportare mediana o media, separando setup della topologia, forward, delta e weight update.

## Report di profiling storico (backprop-mnist, 2026-08-24)

Questo report descrive l'implementazione precedente, basata su adjacency list e propagazione ricorsiva. I risultati non sono direttamente confrontabili con la versione corrente a partition/dense block; resta come traccia della procedura Callgrind.

Contesto: si sospettava che `BackpropRule::learn()` (in `backprop-rule.h`) fosse il collo di
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
   `backprop-mnist.cpp` (modifica temporanea, poi rimossa).
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

La classe `BiasNeuron` è stata rimossa (rispetto a main): il bias è ora gestito tramite `NNet::add_bias(...)`, indipendente dai dati di training.
