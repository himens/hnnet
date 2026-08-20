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

## Priorita alta

- [ ] **Introdurre stato persistente per la learning rule durante il training**
  - `NNet::train()` copia la `LearningRule` per valore e la usa per riferimento non-const per tutta la sessione, quindi lo stato può già persistere tra epoche/sample.
  - Manca però l'uso pratico di questa possibilità: implementare momentum e/o learning rate adattivo in `BackpropRule` per sfruttarla.

## Priorita media

- [ ] **Rendere espliciti input, output e sorgenti della rete**
  - Attualmente `NNet` identifica gli input come neuroni senza connessioni entranti e gli output come neuroni senza connessioni uscenti.
  - Questa deduzione diventa ambigua in reti con rami, neuroni ausiliari o connessioni particolari.
  - Valutare API esplicite per registrare i neuroni di input e output oppure introdurre concetti distinti per source e output neuron.

- [ ] **Gestire il bias come parte della rete, non come dato di training**
  - La classe `BiasNeuron` è stata rimossa, ma il problema di fondo resta: in `backprop-gate.cpp` i bias sono ancora neuroni di input a tutti gli effetti, e richiedono di aggiungere valori costanti (1) a ogni `TrainingSample.inputs` (es. `Data<real_t, 7>` per 2 input reali + 5 bias).
  - Lega il dataset alla topologia interna e rende i dati di input artificialmente più grandi.
  - Introdurre una gestione del bias indipendente dai campioni di addestramento.

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

La classe `BiasNeuron` è stata rimossa (rispetto a main), ma il bias è ancora modellato con neuroni ordinari collegati come input impliciti: il problema descritto sopra (bias non indipendente da `TrainingData`) non è ancora risolto, solo la classe dedicata è sparita.
