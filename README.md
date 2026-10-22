# hNNet

hNNet is a small C++26 framework for building neural networks. Its core separates network structure, signal propagation, and learning strategies:

- `Topology` assembles neurons and connections before network construction
- `NNet` acquires those data, validates connections, and prepares weights, partitions, and input/output/bias metadata; `DAGNet` adds cycle validation and topologically ordered propagation
- `DenseForwardNet` builds fully connected feed-forward networks from a sequence of layer descriptions
- learning rules such as `rule::Perceptron` and `rule::Backprop` plug into `DAGNet::train(...)`

`Neuron` is a concrete type. Its role (`input`, `hidden`, `output`, or `bias`) and activation function are selected when it is created. The built-in learning rules are examples of strategies that operate on the network without making the neuron type depend on a particular learning algorithm.

## Requirements

- CMake 4.4 or newer
- g++ 16
- C++26
- OpenMP

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

### AND gate

```bash
./build/bin/hnnet-perceptron-gate
```

This example trains a network with 2 inputs, a separate bias neuron, and 1 output on the AND truth table. Bias values are not part of the input samples.

### Letter classifier

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

### XOR with backpropagation

```cpp
#include <array>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense_forward_net.h"
#include "hnnet/rules/backprop/backprop.h"
#include "hnnet/rules/backprop/omp/backend.h"
#include "hnnet/rules/backprop/omp/optimizers/momentum.h"

int main() {
    using namespace hnnet;
    net::DenseForwardNet gate{
        net::DenseLayer{2, NeuronType::input, activation::Identity{}},
        net::DenseLayer{4, NeuronType::hidden, activation::Sigmoid{}, true},
        net::DenseLayer{1, NeuronType::output, activation::Sigmoid{}, true}
    };

    const std::array<input_vector_t, 4> inputs{{
        {1, 1}, {1, 0}, {0, 1}, {0, 0}
    }};
    const std::array<output_vector_t, 4> targets{{
        {0}, {1}, {1}, {0}
    }};
    gate.train(
        inputs,
        targets,
        rule::Backprop{
            backprop::omp::Backend<backprop::omp::optimizer::Momentum>{0.2},
            loss::MSE{},
            1
    });
}
```

Build and run the corresponding example with:

```bash
cmake --build build --target hnnet-backprop-gate
./build/bin/hnnet-backprop-gate
```

### MNIST classifier

`hnnet-backprop-mnist` trains a `784-512-512-512-10` network. It expects the CSV files `data/mnist/mnist_train.csv` and `data/mnist/mnist_test.csv` from the project root; each row contains a label followed by 784 pixel values. The example uses up to 60,000 training and 10,000 test samples.

```bash
./build/bin/hnnet-backprop-mnist
```

## Project structure

- `include/`: library headers
    - `hnnet/types.h`: numeric aliases and range concepts for data and datasets
    - `hnnet/activation.h`, `hnnet/neuron.h`: activation interface and neuron definition
    - `hnnet/topology.h`: `Connection`, the mutable `Topology` container, and neuron/connection construction helpers
    - `hnnet/loss_type.h`, `hnnet/learning_rule_type.h`: concepts used by losses and learning rules
    - `hnnet/nnet.h`: base network storage, `NNet::State`, training loop, and inference API
    - `hnnet/activations/`, `hnnet/losses/`: built-in activation and loss implementations
    - `hnnet/rules/`: learning rules; backpropagation backends and optimizers are under `rules/backprop/`
    - `hnnet/nets/`: `DAGNet` and `DenseForwardNet`
- `examples/`: example implementations built on top of the generic framework
- `data/`: datasets used by the examples
- `scripts/`: helper scripts for generating or processing data

## Network API

### Topology and construction

`Topology` owns public `neurons` and `connections` vectors and provides
`new_neurons()`, `connect()` (Cartesian product), and `zip_connect()` (pairwise).
It is a mutable description, not an executable or necessarily acyclic network.

```cpp
#include <utility>
#include "hnnet/activations/identity.h"
#include "hnnet/nets/dag_net.h"
#include "hnnet/topology.h"

hnnet::Topology topology;
auto inputs = topology.new_neurons(
    2, hnnet::NeuronType::input, hnnet::activation::Identity{});
auto outputs = topology.new_neurons(
    1, hnnet::NeuronType::output, hnnet::activation::Identity{});
topology.connect(inputs, outputs);

hnnet::net::DAGNet net{std::move(topology)};
```

The constructor takes ownership of the vectors. `NNet` validates connection
endpoints and duplicates, sorts connections by receiver/transmitter, builds
partitions, and identifies input, output, and bias neurons. `DAGNet` rejects
cycles and orders the partitions topologically during its own construction.
There is no user-facing finalization step, and constructed networks do not
expose neuron or connection construction helpers.

### Networks and learning rules

1. `NNet`
    - abstract, non-template base class that stores neurons, connections, weights, and partitions
    - provides `update(...)`, `infer(...)`, and the common training loop
    - subclasses implement signal propagation

Input and output samples are contiguous ranges of `real_t`; datasets are contiguous ranges of those samples. The aliases `input_vector_t` and `output_vector_t` are both `std::vector<real_t>`.

2. `DAGNet` and `DenseForwardNet`
    - `DAGNet` can be constructed directly from a `Topology`; it propagates dense layers and remaining sparse partitions in topological order
    - dense candidates are grouped directly by their contiguous source range, then checked for contiguous partitions, receivers, and weights; descriptors are stored in topological order and include single-receiver layers
    - `DenseForwardNet` assembles a `Topology` from `DenseLayer` descriptions and passes it to the `DAGNet` constructor; each requested bias is created before the receivers, keeping it contiguous with the source neurons, and connected with independent weights to every receiver

3. `LearningRule`
    - implements `learn(net, inputs, targets)` and is passed to `net.train(inputs, targets, rule)`
    - allows learning strategies to change independently of the network representation

`rule::Backprop<Backend, Loss>` takes `(backend, loss, batch_size)`. The rule owns the backend by value and requires the generic `hnnet::BackendType` contract plus `backprop::BackendType`, which adds the backpropagation operation.

### Views

`NNet::View` exposes read-only spans of neurons, connections, partitions, and
input/output/bias neuron indices, plus a mutable span of weights for learning.
`DAGNet::View` additionally exposes:

- `layers()`: read-only `DAGNet::Layer` descriptors in topological order;
- `partition_layer_indices()`: one index into `layers()` per partition, or
  `DAGNet::Layer::no_layer` when that partition uses the generic sparse path.

Each layer describes contiguous source and receiver ranges and a weight offset.
Its weight row stride equals `tx_count`, including a connected bias when present.

### Backpropagation backends

The OpenMP backpropagation implementation is in `hnnet/rules/backprop/omp/`. `backprop::omp::Backend<Optimizer>` is a backend templated on its optimizer; it owns the optimizer by value and shares the per-thread state, delta and gradient buffers, batch processing, and forward/backward work. The available optimizers are `backprop::omp::optimizer::Momentum` and `backprop::omp::optimizer::Rprop`, each defined in its own header. Both the rule and backend are statically typed, with no virtual dispatch or runtime optimizer selection.

Optimizers own their hyperparameters and per-weight state and satisfy the `backprop::omp::OptimizerType` concept. The current backend processes samples within a mini-batch in parallel with OpenMP, accumulates gradients per thread, and updates weights at the end of each batch.

The Metal backend in `hnnet/rules/backprop/metal/` accepts `DenseForwardNet`
specifically and uses Metal Performance Shaders matrix multiplication for
forward propagation, backward propagation, and gradients. Bias columns are part
of those products, not separate bias kernels. It supports FP32 computation,
Identity/Sigmoid/ReLU activations, MSE, and Momentum or Rprop optimizers.
Metal requires macOS, the Metal SDK, and Objective-C++ compilation; it is not
built by the current OpenMP example targets. See the
[Metal backend documentation](include/hnnet/rules/backprop/metal/README.md)
for backend composition and platform requirements.

Training currently uses a fixed loss threshold and maximum epoch count and prints progress to standard output. Inference and training use `NNet::State` for signals and weighted sums, separately from the network topology.

## TODO

The main planned improvements are:

- add adaptive optimizers such as Adam or RMSProp
- make training thresholds, epoch limits, and reporting configurable
- share activation and loss formulas between CPU and GPU backends
- establish reproducible benchmarks before further dense-kernel optimizations

See [TODO.md](TODO.md) for the complete list of planned improvements and their priorities.

## Notes

The code is intended as a didactic example of a lightweight, extensible neural-network framework.
