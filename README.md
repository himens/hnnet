# hNNet

hNNet is a small C++26 framework for building neural networks. Its core separates network structure, signal propagation, and learning strategies:

- `Topology` assembles neurons and connections before network construction
- `NNet` acquires those data, validates connections, and prepares weights, partitions, and input/output/bias metadata; `DAGNet` adds cycle validation, topologically ordered partitions, and dense-layer detection
- `DenseForwardNet` builds fully connected feed-forward networks from a sequence of layer descriptions
- update backends such as `dag::update::omp::Backend` and `dense::update::omp::Backend` propagate signals through a network view
- learning rules such as `rule::Perceptron` and `rule::Backprop` plug into `train(...)`

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
- if g++ 16 is not the default C++ compiler, add `-DCMAKE_CXX_COMPILER=g++-16` to the configure command
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

1. reads the training files from the `data/letters/` folder
2. creates a network with an input layer and an output layer
3. trains the model on letter examples
4. tries to classify noisy letters

Classifier data representation details:

- each letter is a `9x7` pixel grid
- `#` pixels are encoded as `+1`, `.` pixels as `-1`
- output labels are bipolar too (`+1` for expected letter, `-1` for the others)

> The letter files are not part of the repository: generate them with `python3 scripts/generate_letters.py`, then run the classifier, both from the project root, so that the paths `data/letters/*.txt` are resolved properly.

### XOR with backpropagation

```cpp
#include <array>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense/dense_forward_net.h"
#include "hnnet/rules/backprop/backprop.h"
#include "hnnet/rules/backprop/omp/backend.h"
#include "hnnet/rules/backprop/omp/optimizers/momentum.h"

int main() {
    using namespace hnnet;
    net::DenseForwardNet gate{
        dense::Layer{2, NeuronType::input, activation::Identity{}},
        dense::Layer{4, NeuronType::hidden, activation::Sigmoid{}, true},
        dense::Layer{1, NeuronType::output, activation::Sigmoid{}, true}
    };

    const std::array<std::array<real_t, 2>, 4> inputs{{
        {1, 1}, {1, 0}, {0, 1}, {0, 0}
    }};
    const std::array<std::array<real_t, 1>, 4> targets{{
        {0}, {1}, {1}, {0}
    }};
    gate.train(
        TrainingData{inputs, targets},
        rule::Backprop{
            backprop::omp::Backend<backprop::omp::optimizer::Momentum>{0.2_real},
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

`hnnet-backprop-mnist` trains a `784-128-10` sigmoid network with the OpenMP backend, using mini-batches of 32 samples, and reports the classification efficiency on the training and test sets through `infer(...)`. When compiled with `METAL` defined, it trains a deeper ReLU network (seven hidden layers of 1024 neurons) with the Metal backend instead. It expects the CSV files `data/mnist/mnist_train.csv` and `data/mnist/mnist_test.csv` from the project root; each row contains a label followed by 784 pixel values. The files can be downloaded with `python3 scripts/download_mnist.py`. The example uses up to 60,000 training and 10,000 test samples.

```bash
./build/bin/hnnet-backprop-mnist
```

## Project structure

- `include/`: library headers
    - `hnnet/types.h`: numeric aliases, range concepts for data and datasets, and `TrainingData`
    - `hnnet/activation.h`, `hnnet/neuron.h`: activation interface and neuron definition
    - `hnnet/topology.h`: `Connection`, the mutable `Topology` container, and neuron/connection construction helpers
    - `hnnet/loss_type.h`, `hnnet/learning_rule_type.h`, `hnnet/backend_type.h`, `hnnet/update_backend_type.h`: concepts used by losses, learning rules, and backends
    - `hnnet/nnet.h`: base network storage, `NNet::View`, `NNet::State`, training loop, and inference API
    - `hnnet/activations/`, `hnnet/losses/`: built-in activation and loss implementations
    - `hnnet/rules/`: learning rules; backpropagation backends and optimizers are under `rules/backprop/`
    - `hnnet/nets/dag/`, `hnnet/nets/dense/`: `DAGNet` and `DenseForwardNet`, each with its update backends under `update/`
- `examples/`: example implementations built on top of the generic framework
- `data/`: datasets used by the examples
- `scripts/`: helper scripts for generating or processing data
- `utils/`: small helper library (`utils::utils`) with math, random, and timer utilities

## Network API

### Topology and construction

`Topology` owns public `neurons` and `connections` vectors and provides
`new_neurons()`, `connect()` (Cartesian product), and `zip_connect()` (pairwise).
It is a mutable description, not an executable or necessarily acyclic network.

```cpp
#include <utility>
#include "hnnet/activations/identity.h"
#include "hnnet/nets/dag/dag_net.h"
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
    - non-template base class that stores neurons, connections, weights, and partitions
    - provides `train(data, rule)` and `infer(inputs, backend)`; signal propagation is delegated to update backends
    - exposes its data to learning rules and backends only through a `View`

Input and output samples are contiguous ranges of `real_t`; datasets are contiguous ranges of those samples, and `dataset_vector_t` is `std::vector<std::vector<real_t>>`. `TrainingData{inputs, targets}` bundles the two datasets passed to `train(...)`.

2. `DAGNet` and `DenseForwardNet`
    - `DAGNet` can be constructed directly from a `Topology`; it orders the partitions topologically and detects dense layers
    - dense candidates are grouped directly by their contiguous source range, then checked for contiguous partitions, receivers, and weights; descriptors are stored in topological order and include single-receiver layers
    - `DenseForwardNet` assembles a `Topology` from `dense::Layer` descriptions and passes it to the `DAGNet` constructor; each requested bias is created before the receivers, keeping it contiguous with the source neurons, and connected with independent weights to every receiver
    - `dag::update::omp::Backend` propagates dense layers and remaining sparse partitions in topological order; `dense::update::omp::Backend` provides dense-network inference through it, and Metal provides the corresponding `dense::update::metal::Backend`

Inference takes a dataset and an update backend, for example `net.infer(inputs, dense::update::omp::Backend{})`, and returns one output vector per input sample.

3. `LearningRule`
    - implements `init(view, data)`, `learn()` (one epoch; returns `true` when training has converged), and `apply(weights)`, and is passed to `net.train(TrainingData{inputs, targets}, rule)`
    - allows learning strategies to change independently of the network representation

`rule::Perceptron` takes a learning rate. It works on any `NNet::View`, runs its own forward pass over the connections without an update backend, and is meant for single-layer networks.

`rule::Backprop<Backend, Loss>` takes `(backend, loss, batch_size)`. The rule owns the backend by value and requires the generic `hnnet::BackendType` contract plus `backprop::BackendType`, which adds the backpropagation operation.

### Views

`NNet::View` exposes read-only spans of neurons, connections, weights, partitions,
and input/output/bias neuron indices. Learning rules and backends work on their
own copy of the weights and hand the result back through `apply(weights)`.
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

Training currently uses a fixed loss threshold and maximum epoch count and prints progress to standard output. The OpenMP backends and `rule::Perceptron` use `NNet::State` for signals and weighted sums, separately from the network topology.

## TODO

The main planned improvements are:

- add adaptive optimizers such as Adam or RMSProp
- make training thresholds, epoch limits, and reporting configurable
- share activation and loss formulas between CPU and GPU backends
- establish reproducible benchmarks before further dense-kernel optimizations

See [TODO.md](TODO.md) for the complete list of planned improvements and their priorities.

## Notes

The code is intended as a didactic example of a lightweight, extensible neural-network framework.
