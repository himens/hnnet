# TODO

## Priorita alta

- [ ] **Disaccoppiare il backpropagation dal tipo concreto del neurone**
  - In `BackpropNeuron` la propagazione usa `dynamic_cast<BackpropNeuron*>`.
  - Questo vincola una rete a usare la stessa specializzazione `BackpropNeuron<Activation>` nei neuroni collegati.
  - Introdurre nella classe base `Neuron` un'interfaccia per ricevere un delta, così la propagazione dipende dal contratto comune e non dal tipo concreto.
  - Rendere possibile una rete con attivazioni diverse nei vari layer, ad esempio `ReLU -> Sigmoid -> Linear`.

- [ ] **Separare il calcolo dei gradienti dall'aggiornamento dei pesi**
  - Attualmente ogni neurone aggiorna immediatamente i pesi durante la propagazione del delta.
  - Separare raccolta del gradiente e aggiornamento permetterebbe di introdurre batch, gradient accumulation, momentum e optimizer diversi.
  - Rendere il learning rate configurabile invece di mantenerlo hardcoded dentro `BackpropNeuron`.

## Priorita media

- [ ] **Rendere espliciti input, output e sorgenti della rete**
  - Attualmente `NNet` identifica gli input come neuroni senza connessioni entranti e gli output come neuroni senza connessioni uscenti.
  - Questa deduzione diventa ambigua in reti con rami, neuroni ausiliari o connessioni particolari.
  - Valutare API esplicite per registrare i neuroni di input e output oppure introdurre concetti distinti per source e output neuron.

- [ ] **Gestire il bias come parte della rete, non come dato di training**
  - L'attuale soluzione usa neuroni aggiuntivi e richiede di aggiungere valori costanti a `TrainingData`.
  - È stata utile per testare il codice, ma rende i dati di input artificialmente più grandi e lega il dataset alla topologia interna.
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

La classe `BiasNeuron` è considerata superflua e va eliminata. Il bias non dovrebbe richiedere l'aggiunta di elementi artificiali a `TrainingData`, ma essere modellato come proprietà della rete o del layer.
