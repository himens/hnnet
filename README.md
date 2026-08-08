# hNNet

hNNet is a small C++ library for building simple neural networks based on perceptrons. The project also includes two practical examples:

- a simple AND gate built with a perceptron network
- a letter classifier that uses data from the data/ folder

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

This example creates a network with 3 inputs and 1 output, trains the model on a few AND gate examples, and verifies the behavior of the network.

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
  - `hnnet.h`: basic types such as `Data`
  - `hnnet-neuron.h`: definition of the `Neuron` class
  - `hnnet-nnet.h`: implementation of the `NNet` network
- `examples/perceptron/`: usage examples
- `data/`: datasets used by the examples
- `scripts/`: helper scripts for generating or processing data

## Basic library usage

Here is a minimal example of how to create and train a network:

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

The main functions are:

- `new_neuron<T>()`: creates a single neuron
- `new_neurons<T>(n)`: creates `n` neurons
- `connect(...)`: connects neurons to each other
- `train(...)`: trains the network on the provided data
- `infer(...)`: performs inference on new inputs

## Notes

The code is intended as a didactic example and shows a simple implementation of a perceptron-based neural network, not a production-ready library.
