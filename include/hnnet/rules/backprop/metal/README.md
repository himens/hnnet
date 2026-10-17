# Metal Backprop Backend

This directory contains the Objective-C++ host implementation and Metal Shading
Language kernels. No OpenMP code is used by the Metal backend.

## Build Requirements

- macOS with the Metal SDK and a compatible GPU.
- Compile the translation unit that constructs the Metal Backprop rule as
  Objective-C++ (`.mm`, or `-x objective-c++`) with ARC (`-fobjc-arc`). The
  current hnnet headers require C++26; `-std=c++2c` is Clang's draft spelling.
- Link the Metal and Foundation frameworks (`-framework Metal -framework Foundation`).
- Keep `activations.h` and `kernels.metal` available at runtime. The default path
  is the directory containing `backend.h`, as recorded by `__FILE__`. If that
  path is relative, it is resolved against the process working directory. For
  relocated installations, call `set_kernel_directory(path)` before `init` or
  before moving the backend into the rule.

The library compiles its shader source during `init`; a precompiled metallib and
changes to the project CMake files are not required by this implementation.

## Compile a Metal Example

Run this from the repository root, replacing `path/to/backprop_metal.mm` with
the `.mm` file containing the Metal example:

```bash
mkdir -p build/bin
xcrun --sdk macosx clang++ \
    -std=c++2c -stdlib=libc++ -fobjc-arc -x objective-c++ \
    -Iinclude -Iutils/include \
    path/to/backprop_metal.mm \
    -framework Foundation -framework Metal \
    -o build/bin/backprop-metal
```

For example, if the source is `examples/backprop/src/backprop_metal.mm`, use
that path in the command. The existing `backprop_mnist.cpp` is the OpenMP example,
not a Metal translation unit. Run the resulting executable from the repository
root so the default kernel directory recorded from `__FILE__` resolves correctly.
No `-fopenmp` flag is needed for this Metal backend.

## Composition

```cpp
#include "hnnet/rules/backprop/backprop.h"
#include "hnnet/rules/backprop/metal/backend.h"
#include "hnnet/rules/backprop/metal/optimizers/rprop.h"

using namespace hnnet;

auto training_rule = rule::Backprop{
    backprop::metal::Backend<backprop::metal::optimizer::Rprop>{1.2_real, 0.5_real},
    loss::MSE{},
    64
};

net.train(inputs, targets, training_rule);
```

Momentum is available in `optimizers/momentum.h`, with constructor parameters
`learning_rate` and optional `momentum`. The backend owns the selected optimizer
by value. Its buffers are initialized and reset together with the network in
`init(net)`.

## Supported Operations

- Identity, Sigmoid (including its `sigma` parameter), and ReLU, with derivatives.
- MSE loss (`0.5 * error * error`). Other loss types are not accepted by `backprop`.
- Momentum and iRprop-, with persistent per-weight state on the device.
- Acyclic networks described by `net::DAGNet::View`, including bias neurons,
  skip connections and sparse connections. Input and bias neurons must not have
  incoming connections. Unknown activations and unsupported topologies are
  rejected explicitly rather than evaluated on the CPU.

All GPU tensors and arithmetic use FP32. Host `real_t` is unchanged: data and
weights are converted on upload and weights are converted back on download.
Nonfinite inputs, targets, initial weights and activation parameters are rejected.
Results need not be bitwise identical to OpenMP, particularly with host FP64 or
different summation orders. iRprop- may take a different trajectory near zero
gradients because it depends on their signs.

## Execution

`init` builds incoming/outgoing adjacency lists and topological levels, compiles
the kernels and initializes optimizer buffers. The dense-block optimization of
OpenMP is not required: the GPU executes the equivalent connection graph.

`to_device` uploads the dataset and current host weights. It is intended to be
called once per epoch, as in the existing Backprop rule. Intermediate weights
remain on the GPU between batches. Each `backprop` call encodes seeding,
forward levels, MSE, backward levels, the batch-mean gradient and one optimizer
update. Separate encoders order dependent levels without floating-point atomics.
The call waits for completion and returns the sum of pre-update sample losses;
only this final scalar accumulation runs on the host. `to_host` copies updated
weights into the network.

Scratch storage grows with the largest batch seen: three FP32 arrays of
`batch_size * neuron_count`, plus losses and the connection gradient. Full-batch
training can therefore require substantial memory. The initial implementation
uses shared Metal buffers and synchronous batch completion, prioritizing a
verifiable contract over transfer/dispatch optimizations. Tensor indexing is
limited to 32 bits and allocations are checked against the device buffer limit.

Metal resource handles are retained by ARC. As with other handle-based APIs,
copies of an already initialized backend share GPU buffers; do not use such
copies concurrently. Construct independent backends for independent training
sessions. The normal training path copies the rule before `init`, which creates
its device resources for that session.

## Local Verification

This code has not been compiled with the Apple SDK or executed on a GPU in the
Linux development container. Before running MNIST, verify on macOS:

- A small Identity network with known weights, comparing loss, gradients and
  several Momentum/Rprop steps against the CPU backend within FP32 tolerance.
- Sigmoid with default and non-default `sigma`, and ReLU on both sides of zero.
- Bias, branching/skip connections, batch size one and a partial final batch.
- Reset through `init`, upload/download round trips and independent backends.
- Invalid sample shapes, unsupported activations and missing shader files.

Enable Metal API validation and shader validation during these checks. No changes
outside this directory were made to integrate or enable the backend.