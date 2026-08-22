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

## Nota

La classe `BiasNeuron` è stata rimossa (rispetto a main): il bias è ora gestito tramite `NNet::add_bias(...)`, indipendente dai dati di training.
