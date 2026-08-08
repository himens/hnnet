# hNNet

hNNet è una piccola libreria C++ per costruire reti neurali semplici basate su perceptroni. Il progetto include anche due esempi pratici:

- un semplice gate AND costruito con una rete percettrone
- un classificatore di lettere che usa dati presenti nella cartella data/

## Requisiti

- CMake 3.16 o superiore
- g++ 16
- C++23

Il progetto è già configurato per usare il compilatore `/usr/bin/g++-16`.

## Compilazione

Dalla radice del repository esegui:

```bash
cmake -S . -B build
cmake --build build
```

Gli eseguibili vengono generati nella cartella `build/bin`.

## Esecuzione degli esempi

### 1. Esempio gate AND

```bash
./build/bin/hnnet-perceptron-gate
```

Questo esempio crea una rete con 3 ingressi e 1 uscita, addestra il modello su alcuni esempi del gate AND e verifica il comportamento del network.

### 2. Classificatore di lettere

```bash
./build/bin/hnnet-perceptron-classifier
```

Questo esempio:

1. legge i file di training dalla cartella `data/`
2. crea una rete con un layer di input e uno di output
3. addestra il modello su esempi di lettere
4. prova a classificare lettere rumorose

> Per far funzionare il classificatore correttamente, esegui il comando dalla radice del progetto, così che i percorsi `data/letters_*.txt` vengano risolti correttamente.

## Struttura del progetto

- `include/`: header della libreria
  - `hnnet.h`: tipi base come `Data`
  - `hnnet-neuron.h`: definizione della classe `Neuron`
  - `hnnet-nnet.h`: implementazione della rete `NNet`
- `examples/perceptron/`: esempi di utilizzo
- `data/`: dataset usati dagli esempi
- `scripts/`: script utili per generare o elaborare dati

## Uso base della libreria

Ecco un esempio minimo di come creare e addestrare una rete:

```cpp
#include "hnnet-nnet.h"

using Net = hNNet::NNet<3, 1>;
Net net;

const auto inputs = net.new_neurons<Perceptron::Neuron>(3);
const auto output = net.new_neuron<Perceptron::Neuron>();

net.connect(inputs, output);

std::vector<Net::TrainingData> samples = {
    {{1, 1, 1}, {1}},
    {{1, 0, 1}, {-1}},
    {{0, 1, 1}, {-1}},
    {{0, 0, 1}, {-1}}
};

net.train(samples);
```

Le funzioni principali sono:

- `new_neuron<T>()`: crea un singolo neurone
- `new_neurons<T>(n)`: crea `n` neuroni
- `connect(...)`: collega neuroni tra loro
- `train(...)`: addestra la rete sui dati forniti
- `infer(...)`: esegue l'inferenza su nuovi input

## Note

Il codice è pensato come esempio didattico e mostra un'implementazione semplice di una rete neurale basata su perceptroni, non una libreria pronta per uso industriale.
