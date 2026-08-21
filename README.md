# hNNet

hNNet is a small C++ framework for building neural networks in a generic way. The core idea is not to hard-code a specific model but to provide:

- a generic neural network, `NNet`, whose template parameters define the type and size of the input and output data
- a single concrete `Neuron` class, whose type (`input`, `hidden`, `output`, `bias`) and activation function are set when it is created
- a `LearningRule` strategy (e.g. `Builtin::PerceptronRule`, `Builtin::BackpropRule`) that plugs into `NNet::train(...)` without requiring virtual dispatch on the neuron

The perceptron and backprop rules in `include/hnnet/builtin` are only two concrete learning strategies that can be plugged into the generic framework.

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
    - `hnnet/activation.h`: `Activation` strategies (identity, step, sigmoid, ...)
    - `hnnet/neuron.h`: definition of the `Neuron` class
    - `hnnet/learning-rule.h`: `LearningRuleType` concept used by `NNet::train(...)`
    - `hnnet/nnet.h`: implementation of the generic `NNet` network
    - `hnnet/builtin/`: built-in learning rules (`PerceptronRule`, `BackpropRule`)
- `examples/`: example implementations built on top of the generic framework
- `data/`: datasets used by the examples
- `scripts/`: helper scripts for generating or processing data

## How the library is intended to be used

The library is designed around three main pieces:

1. `NNet<InputData, OutputData>`
    - defines the network input/output contract through `Data` types
    - manages neuron creation and connections
    - provides `train(...)` and `infer(...)` operations

`InputData` and `OutputData` must satisfy `DataType`, so in practice you pass aliases based on `Data<ValueType, Size>`.

2. `Neuron`
    - a single concrete class: its `NeuronType` (`input`, `hidden`, `output`, `bias`) is set at creation and used by `NNet` to identify input/output neurons
    - owns an `Activation` strategy (e.g. `SigmoidActivation`, `PerceptronActivation`) used to convert weighted sums into output signals

3. `LearningRule` (e.g. `Builtin::PerceptronRule`, `Builtin::BackpropRule`)
    - implements `learn(net, targets)` and is passed to `NNet::train(...)`
    - lets you swap the learning algorithm without changing `Neuron` or `NNet`

## Basic usage example

Here is a minimal example of how to create and train a network (AND gate with a perceptron rule):

```cpp
#include "hnnet/nnet.h"
#include "hnnet/builtin/perceptron-rule.h"

using namespace hNNet;
using Gate = NNet<Data<int_t, 2>, Data<int_t, 1>>;
Gate gate;

const auto input_layer  = gate.new_neurons(2, NeuronType::input);
const auto output_layer = gate.new_neurons(1, NeuronType::output, PerceptronActivation{});
gate.connect(input_layer, output_layer);
gate.add_bias(output_layer);

std::vector<Gate::TrainingSample> samples = {
    {{1, 1},  {1}},
    {{1, 0}, {-1}},
    {{0, 1}, {-1}},
    {{0, 0}, {-1}}
};

gate.train(samples, Builtin::PerceptronRule{1.0});
```

The main operations are:

- `new_neurons(n, type, activation)`: creates `n` neurons of the given `NeuronType` (and optional `Activation`, identity by default)
- `connect(tx, rx)`: connects neurons to each other; each side can be a single `Neuron` or a range of neurons (all tx-rx pairs are connected)
- `add_bias(neurons)`: creates and connects one bias neuron (constant signal 1.0) per neuron in the given range
- `train(samples, rule)`: trains the network on the provided data using the given `LearningRule`
- `infer(data)`: performs inference on new inputs

During `train(...)`, the current implementation prints epoch progress and, when converged, a short summary with learned weights, elapsed time, and total epochs.

## TODO

The main planned improvements are:

- give the learning rules persistent state across epochs (momentum, adaptive learning rate)
- reduce the implicit dependency on propagation order
- add a layer/model abstraction and configurable training parameters

See [TODO.md](TODO.md) for the complete list of planned improvements and their priorities.

## Notes

The code is intended as a didactic example and shows a lightweight, extensible neural-network framework.
