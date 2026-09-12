# hNNet

hNNet is a small C++ framework for building neural networks in a generic way. The core idea is not to hard-code a specific model but to provide:

- a generic neural network, `NNet`, whose template parameters define the type and size of the input and output data
- a single concrete `Neuron` class, whose type (`input`, `hidden`, `output`, `bias`) and activation function are set when it is created
- a `LearningRule` strategy (e.g. `Builtin::PerceptronRule`, `Builtin::BackpropRule`) that plugs into `NNet::train(...)` without requiring virtual dispatch on the neuron

The perceptron and backprop rules in `include/hnnet/builtin` are only two concrete learning strategies that can be plugged into the generic framework.

## Requirements

- CMake 4.4 or newer
- g++ 16
- C++26

## Build

From the repository root, run:

```bash
cmake -S . -B build
cmake --build build
```

The executables are generated in the `build/bin` folder.

Notes on current build configuration:

- use `-DCMAKE_BUILD_TYPE=Release` for optimized builds
- on GNU/Clang toolchains CMake enables `-march=native` (or `-mcpu=native` on ARM)
- examples link against the interface target `hnnet::hnnet`

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
    - `hnnet/activation.h`: `ActivationType` concept
    - `hnnet/neuron.h`: definition of the `Neuron` class
    - `hnnet/loss.h`: `LossType` concept used by learning rules
    - `hnnet/learning_rule.h`: `LearningRuleType` concept used by `NNet::train(...)`
    - `hnnet/nnet.h`: implementation of the generic `NNet` network and `NNetState`
    - `hnnet/builtin/`: built-in activations (`activations.h`), losses (`losses.h`), learning rules (`perceptron_rule.h`, `backprop_rule.h`) and the `DenseForwardNet` helper (`dense_forward_net.h`)
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

Here is a minimal example of how to create and train a network (XOR gate with a backprop rule and a hidden layer):

```cpp
#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/dense_forward_net.h"
#include "hnnet/builtin/backprop_rule.h"

using namespace hNNet;
using Gate = Builtin::DenseForwardNet<Data<real_t, 2>, Data<real_t, 1>>;
Gate gate{
    Builtin::Layer{2, NeuronType::input,  Builtin::IdentityActivation{}},
    Builtin::Layer{4, NeuronType::hidden, Builtin::SigmoidActivation{}, true},
    Builtin::Layer{1, NeuronType::output, Builtin::SigmoidActivation{}, true}
};

std::vector<Gate::TrainingData> samples{
    {{1, 1}, {0}},
    {{1, 0}, {1}},
    {{0, 1}, {1}},
    {{0, 0}, {0}}
};

gate.train(samples, Builtin::BackpropRule{0.2});
```

The main operations are:

- `DenseForwardNet<InputData, OutputData>{Layer{...}, ...}`: builds a fully-connected feed-forward network from a runtime list of `Layer` descriptions (size, `NeuronType`, activation, optional bias)
- `new_neurons(n, type, activation)` / `connect(tx, rx)`: lower-level primitives used internally by `DenseForwardNet` if you need to build a custom topology by hand
- `train(samples, rule)`: trains the network on the provided data using the given `LearningRule` for a whole epoch at a time
- `infer(data)`: performs inference on new inputs

`Builtin::BackpropRule<Loss, Optimizer>` takes `(learning_rate, optimizer = {}, batch_size = 1, loss = {})`: samples are split into mini-batches, each mini-batch is processed sequentially by one thread (multiple mini-batches run in parallel across threads), and weights are updated once per epoch from the accumulated gradient. `Builtin::SGDMomentum` is the built-in optimizer, holding only the momentum-specific state (the learning rate lives on `BackpropRule`, not on the optimizer, so it stays generic across optimizers).

Internally, `NNet` compiles the connection list into receiver partitions in topological order. Compatible contiguous partitions are grouped into dense blocks, so their forward pass uses sequential weight and signal buffers. Per-sample transient state (signals and weighted sums) lives in `NNetState`, separate from the network topology, so it can be duplicated per thread for parallel training.

During `train(...)`, the current implementation prints epoch progress and, when converged, a short summary with elapsed time and total epochs.

## TODO

The main planned improvements are:

- add adaptive optimizers such as Adam or RMSProp
- add a layer/model abstraction beyond `DenseForwardNet` and configurable training parameters
- evaluate a GPU backend (e.g. Metal) in addition to the current OpenMP-based CPU parallelism

See [TODO.md](TODO.md) for the complete list of planned improvements and their priorities.

## Notes

The code is intended as a didactic example and shows a lightweight, extensible neural-network framework.
