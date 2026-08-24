# TODO

## Fatto (rispetto a main)

- [x] **Disaccoppiare il backpropagation dal tipo concreto del neurone**
  - `Neuron` non è più templata sull'attivazione: `BackpropNeuron`/`PerceptronNeuron` sono stati rimossi, l'attivazione è un `Activation` (variant) tenuto per valore in `Neuron`.
  - La logica di apprendimento è stata estratta in `LearningRule` (`BackpropRule`, `PerceptronRule`) che opera su `NNet::View`/`Neuron` tramite un'interfaccia comune, senza `dynamic_cast`.
  - Reti con attivazioni diverse per neurone sono già possibili.

- [x] **Separare il calcolo dei gradienti dall'aggiornamento dei pesi**
  - In `BackpropRule::learn()` i delta vengono calcolati e retro-propagati in un primo passaggio, l'aggiornamento dei pesi avviene in un secondo ciclo separato.
  - Il learning rate è ora un parametro del costruttore di `BackpropRule`/`PerceptronRule`, non più hardcoded.
  - Ancora da fare: batch/gradient accumulation e optimizer alternativi (momentum, adaptive rate) non sono ancora implementati, l'architettura lo permette ma manca lo stato persistente tra epoche (vedi sotto).

- [x] **Rendere espliciti input, output e sorgenti della rete**
  - I neuroni sono ora etichettati con `NeuronType` (`input`, `hidden`, `output`, `bias`), assegnato esplicitamente alla creazione con `new_neurons(...)`.
  - `NNet::input_neurons()`/`output_neurons()` selezionano i neuroni per tipo e validano che il conteggio corrisponda a `input_size`/`output_size`.
  - Non c'è più alcuna deduzione basata su connessioni entranti/uscenti.

- [x] **Gestire il bias come parte della rete, non come dato di training**
  - `NNet::add_bias(neurons)` crea un neurone di bias per ciascun neurone del range passato, lo collega e lo attiva con segnale costante 1.0.
  - Il bias non è più parte di `TrainingSample.inputs`: la topologia della rete non influenza più la forma dei dati di training.

- [x] **Unificare gli overload di `connect`**
  - `NNet::connect(tx, rx)` accetta ora, per ciascun lato, sia un singolo `Neuron` sia un `NeuronRange`, tramite un'unica funzione template invece di quattro overload distinti.
  - Internamente entrambi gli argomenti vengono uniformati a una view di `Neuron*` per riutilizzare la stessa logica di controllo/creazione delle connessioni.

## Priorita alta

- [ ] **Introdurre stato persistente per la learning rule durante il training**
  - `NNet::train()` copia la `LearningRule` per valore e la usa per riferimento non-const per tutta la sessione, quindi lo stato può già persistere tra epoche/sample.
  - Manca però l'uso pratico di questa possibilità: implementare momentum e/o learning rate adattivo in `BackpropRule` per sfruttarla.

## Priorita media

- [ ] **Ridurre la dipendenza implicita dall'ordine di propagazione**
  - Il conteggio dei delta ricevuti presume che ogni connessione produca esattamente un delta per ciclo.
  - Documentare e validare le assunzioni su rete aciclica, connessioni stabili e propagazione completa.
  - Gestire esplicitamente eventuali topologie non supportate.

## Priorita bassa

- [ ] **Introdurre un'astrazione per layer o modelli di rete**
  - Oggi l'utente deve creare e collegare manualmente ogni gruppo di neuroni.
  - Aggiungere helper per costruire layer densi e collegamenti tra layer ridurrebbe il codice ripetitivo senza nascondere il grafo quando serve controllo fine.

- [ ] **Migliorare la configurazione dell'addestramento**
  - Rendere configurabili soglia di errore, numero massimo di epoche, learning rate e strategia di aggiornamento.
  - Valutare metriche e callback separati dal logging diretto su stdout.

## Performance (accesso memoria, cache, parallelizzazione)

Principio guida: il progetto punta a un framework generico (topologia a grafo arbitraria, attivazioni per-neurone, bias come neuroni normali) ma anche performante. Il tradeoff va cercato ottimizzando la rappresentazione *interna* usata nel hot path, senza sacrificare la generalità dell'API (`connect()` libero, tipi di neurone, attivazioni miste). Dove la generalità impone un costo strutturale non banale (es. dispatch per-attivazione, stato per-sample condiviso con la rete), va valutato caso per caso se e come disaccoppiarlo, anche a costo di un cambio architetturale più profondo.

- [ ] **Sostituire le adjacency list `vector<vector<index_t>>` con un formato CSR**
  - `_in_connections`/`_out_connections` allocano un vettore separato per neurone: frammentazione e cache-miss ad ogni attraversamento del grafo.
  - Usare un formato compressed-sparse-row (un unico buffer contiguo + vettore di offset) elimina le allocazioni per-neurone e migliora la località.
  - Problema pratico: oggi gli archi vengono aggiunti incrementalmente da `connect()`/`zip_connect()`/`add_bias()` in ordine arbitrario di neurone, mentre un CSR vero è naturalmente immutabile una volta costruito (inserire un arco "in mezzo" richiederebbe spostare tutto il buffer).
  - Approccio pragmatico a due fasi: durante la costruzione accumulare gli archi in una edge-list di appoggio economica (`vector<pair<index_t /*nodo*/, index_t /*iconn*/>>`, append-only, O(1) per inserimento); poi, prima di `train()`/`infer()`, un `compile()` una tantum calcola gli offset per neurone con un prefix-sum e riempie il buffer piatto con un counting sort O(V+E). Il costo di compattazione si paga una sola volta in fase di setup (dove già oggi si paga l'O(n²) del controllo duplicati), non ad ogni epoca/sample.
  - Da adattare anche `View::in_connections`/`out_connections`, che oggi tornano `const vector<index_t>&`, per restituire uno `std::span<const index_t>` sul buffer compilato.

- [ ] **Eliminare la doppia indirezione nell'attraversamento del grafo**
  - Oggi ogni arco richiede: indice in `_in_connections`/`_out_connections` -> lookup in `_connections` -> `(itx, irx, weight)`.
  - Valutare di colocare `(indice/puntatore vicino, weight)` direttamente nelle adjacency list per un solo accesso in memoria per arco.

- [ ] **Rendere O(1) il controllo di connessione duplicata in `connect()`**
  - Attualmente `std::ranges::any_of` scansiona tutte le connessioni esistenti per ogni nuova coppia (tx, rx): O(n²) nella costruzione di layer densi.
  - Sostituire con una hash set su `(itx, irx)`.

- [ ] **Precalcolare e riutilizzare l'ordine topologico di attivazione**
  - L'ordine con cui i neuroni si attivano viene riscoperto a runtime (via conteggio dei segnali ricevuti) ad ogni singolo sample, sia in `broadcast` che in `backprop_error`.
  - Se la topologia è fissa tra un'epoca e l'altra, calcolare l'ordine una sola volta e riusarlo evita il lavoro ripetuto e apre la porta a cicli non ricorsivi.

- [ ] **Convertire `broadcast`/`backprop_error` da ricorsione a iterazione esplicita**
  - La ricorsione con branching dipendente dai dati impedisce vettorizzazione/inlining e rischia stack overflow su reti profonde.
  - Un attraversamento iterativo (coda/worklist, o meglio ancora un ordine topologico precalcolato) sarebbe più prevedibile e veloce.

- [ ] **Ridurre il dispatch dinamico per-neurone sull'attivazione**
  - Ogni `activate()`/`activation_derivative()` passa da `std::visit` su `Neuron::_activation`.
  - Raggruppare i neuroni per tipo di attivazione (o per layer) permetterebbe di applicare la funzione in batch su dati contigui, evitando il dispatch ripetuto.

- [ ] **Separare stato per-sample da topologia/pesi per abilitare il parallelismo**
  - Lo stato di attivazione (`_weighted_sum`, `_signal`, `_number_rx_signals`) vive dentro lo stesso `Neuron` che rappresenta la rete condivisa: impossibile elaborare più sample in parallelo (mini-batch, multi-thread) senza duplicare l'intera rete.
  - Estrarre lo stato per-sample in una struttura separata (indicizzata per neurone) renderebbe il forward/backward pass rientrante e parallelizzabile su sample o thread.

## Stress test / Benchmark

- [ ] **Aggiungere un esempio/benchmark su MNIST con una rete più grande**
  - Gli esempi attuali (gate logici, lettere 9×7) sono troppo piccoli per far emergere i problemi di performance sopra elencati (ricorsione profonda, O(n²) in `connect()`, cache locality delle adjacency list).
  - MNIST (784 input, uno o più hidden layer, 10 output) darebbe una rete di dimensioni realistiche per misurare tempo di setup della topologia, tempo per epoca e per inferenza, prima e dopo ogni ottimizzazione.
  - Utile anche per validare in modo empirico (non solo teorico) l'impatto di ciascuna voce della sezione Performance.
  - Target/baseline di riferimento (letteratura, non specifici di `hnnet`): un MLP 784→hidden→10 arriva tipicamente al 95-98% di accuratezza in 10-30 epoche; su CPU con implementazione a matrici dense (BLAS/SIMD) un'epoca su 60k sample è dell'ordine di 1-3s, quindi training completo in decine di secondi/pochi minuti. `hnnet`, essendo push-based su grafo (dispatch per-arco, `std::visit` per attivazione, niente operazioni matriciali), va misurato: è atteso un gap significativo (probabilmente un ordine di grandezza o più) rispetto a questi tempi, ed è proprio quel gap che il benchmark deve quantificare.

## Report di profiling (backprop-mnist, 2026-08-24)

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
