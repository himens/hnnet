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

## Nota

La classe `BiasNeuron` è stata rimossa (rispetto a main): il bias è ora gestito tramite `NNet::add_bias(...)`, indipendente dai dati di training.
