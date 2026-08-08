# hNNet

hNNet is a small C++ framework for building neural networks in a generic way. The core idea is not to hard-code a specific model but to provide:

- a generic network container, `NNet`, whose template parameters define the size of the input and output data
- a flexible base neuron abstraction, `Neuron`, whose behavior can be customized through virtual hooks
- a simple training and inference loop that works once the network topology and neuron behavior are defined

The perceptron code in the examples is only one concrete implementation of a neuron.

## Requirements

- CMake 3.16 or newer
- g++ 16
- C++23

The project is already configured to use the compiler `/usr/bin/g++-16`.

## Build

From the repository root, run:

```bash
cmake -S . -B build
cmake --build build
```

The executables are generated in the `build/bin` folder.

## Running the examples

### 1. AND gate example

```bash
./build/bin/hnnet-perceptron-gate
```

This example builds a small network with 3 inputs and 1 output, trains it and checks the resulting behavior.

### 2. Letter classifier

```bash
./build/bin/hnnet-perceptron-classifier
```

This example:

1. reads the training files from the `data/` folder
2. creates a network with an input layer and an output layer
3. trains the model on letter examples
4. tries to classify noisy letters

> To make the classifier work correctly, run the command from the project root so that the paths `data/letters_*.txt` are resolved properly.

## Project structure

- `include/`: library headers
  - `hnnet.h`: base types such as `Data`
  - `hnnet-neuron.h`: definition of the generic `Neuron` abstraction
  - `hnnet-nnet.h`: implementation of the generic `NNet` network
- `examples/perceptron/`: example implementations built on top of the generic framework
- `data/`: datasets used by the examples
- `scripts/`: helper scripts for generating or processing data

## How the library is intended to be used

The library is designed around two main pieces:

1. `NNet<SizeInput, SizeOutput>`
   - defines the network shape through template parameters
   - manages neuron creation and connections
   - provides `train(...)` and `infer(...)` operations

2. `Neuron`
   - provides the generic signal propagation and learning infrastructure
   - exposes customizable virtual behavior through methods such as `update(...)` and `activation(...)`
   - allows you to define your own neuron model by deriving from it

In other words, the framework is generic, and the perceptron implementation is just one possible neuron model that can be plugged into it.

## Basic usage example

Here is a minimal example of how to create and train a network using a custom neuron type:

```cpp
#include "hnnet-nnet.h"

struct MyNeuron : public hNNet::Neuron {
    void update(SynapticConn* conn, const hNNet::real_t target) override {
        conn->weight += target * conn->tx->get_signal();
    }

    hNNet::real_t activation(const hNNet::real_t weighted_sum) const override {
        return weighted_sum > 0.0 ? 1.0 : -1.0;
    }
};

using Net = hNNet::NNet<3, 1>;
Net net;

const auto inputs = net.new_neurons<MyNeuron>(3);
const auto output = net.new_neuron<MyNeuron>();

net.connect(inputs, output);

std::vector<Net::TrainingData> samples = {
    {{1, 1, 1}, {1}},
    {{1, 0, 1}, {-1}},
    {{0, 1, 1}, {-1}},
    {{0, 0, 1}, {-1}}
};

net.train(samples);
```

The main operations are:

- `new_neuron<T>()`: creates a single neuron
- `new_neurons<T>(n)`: creates `n` neurons
- `connect(...)`: connects neurons to each other
- `train(...)`: trains the network on the provided data
- `infer(...)`: performs inference on new inputs

## Notes

The code is intended as a didactic example and shows a lightweight, extensible neural-network framework.
