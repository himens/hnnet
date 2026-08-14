# hNNet

hNNet is a small C++ framework for building neural networks in a generic way. The core idea is not to hard-code a specific model but to provide:

- a generic neural network, `NNet`, whose template parameters define the type and size of the input and output data
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

Notes on current build configuration:

- the project uses `/usr/bin/g++-16` and C++23
- on GNU/Clang toolchains, CMake enables `-O3` for C++ compilation
- examples link against the interface target `hnnet::hnnet_lib` (which also links `utility::utility`)

## Running the examples

### 1. AND gate example

```bash
./build/bin/hnnet-perceptron-gate
```

This example builds a small network with 3 inputs (2 + bias) and 1 output, trains it and checks that the result is the one of an AND gate.

### 2. Letter classifier

```bash
./build/bin/hnnet-perceptron-classifier
```

This example:

1. reads the training files from the `data/` folder
2. creates a network with an input layer and an output layer
3. trains the model on letter examples
4. tries to classify noisy letters

Classifier data representation details:

- each letter is a `9x7` pixel grid
- `#` pixels are encoded as `+1`, `.` pixels as `-1`
- output labels are bipolar too (`+1` for expected letter, `-1` for the others)

> To make the classifier work correctly, run the command from the project root so that the paths `data/letters_*.txt` are resolved properly.

## Project structure

- `include/`: library headers
    - `hnnet/types.h`: base types such as `Data`
    - `hnnet/neuron.h`: definition of the generic `Neuron` abstraction
    - `hnnet/nnet.h`: implementation of the generic `NNet` network
- `examples/`: example implementations built on top of the generic framework
- `data/`: datasets used by the examples
- `scripts/`: helper scripts for generating or processing data

## How the library is intended to be used

The library is designed around two main pieces:

1. `NNet<InputData, OutputData>`
    - defines the network input/output contract through `Data` types
   - manages neuron creation and connections
   - provides `train(...)` and `infer(...)` operations

`InputData` and `OutputData` must satisfy `DataType`, so in practice you pass aliases based on `Data<ValueType, Size>`.

2. `Neuron`
   - provides the generic signal propagation and learning infrastructure
    - owns an `Activation` strategy used to convert weighted sums into output signals
    - exposes customizable virtual behavior through methods such as `update(...)`
   - allows you to define your own neuron model by deriving from it

In other words, the framework is generic, and the perceptron implementation is just one possible neuron model that can be plugged into it.

## Basic usage example

Here is a minimal example of how to create and train a network using a custom neuron type:

```cpp
#include "hnnet/nnet.h"
#include "hnnet/builtin/activations.h"

struct MyNeuron : public hNNet::Neuron {
    MyNeuron()
        : hNNet::Neuron(std::make_unique<hNNet::BipolarStepActivation>()) {}

    void learn(const hNNet::real_t error) override {
        static constexpr real_t rate{1.0};
        for (auto &conn : this->get_in_connections()) {
             const auto target = error + this->get_signal();
             conn->weight += rate * target * conn->tx->get_signal();
        }
    }
};

using InputData = hNNet::Data<hNNet::int_t, 3>;
using OutputData = hNNet::Data<hNNet::int_t, 1>;
using Net = hNNet::NNet<InputData, OutputData>;
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

During `train(...)`, the current implementation prints epoch progress and, when converged, a short summary with learned weights, elapsed time, and total epochs.

## TODO

The main planned improvements are:

- decouple backpropagation from the concrete neuron type
- separate gradient calculation from weight updates
- make inputs, outputs, and biases explicit parts of the network
- add configurable training parameters and layer-building helpers

See [TODO.md](TODO.md) for the complete list of planned improvements and their priorities.

## Notes

The code is intended as a didactic example and shows a lightweight, extensible neural-network framework.
