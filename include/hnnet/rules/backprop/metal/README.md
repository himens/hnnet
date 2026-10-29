# Metal Backprop Backend

This directory contains the Objective-C++ host implementation and Metal Shading
Language kernels. No OpenMP code is used by the Metal backend.

## Build Requirements

- macOS with the Metal SDK and a compatible GPU.
- Compile the translation unit that constructs the Metal Backprop rule as
  Objective-C++ (`.mm`, or `-x objective-c++`) with ARC (`-fobjc-arc`). The
  current hnnet headers require C++26; `-std=c++2c` is Clang's draft spelling.
- Link Metal, MetalPerformanceShaders and Foundation (`-framework Metal
  -framework MetalPerformanceShaders -framework Foundation`).
- Keep the local `kernels.metal` and the shared forward shaders
  [`activations.h`](../../../nets/dense_forward/update/metal/activations.h)
  and [`kernels.metal`](../../../nets/dense_forward/update/metal/kernels.metal)
  available at runtime. Default paths are recorded by `__FILE__` and relative
  paths are resolved against the process working directory. For relocated
  installations, call `set_kernel_directory(path)` for backprop shaders and
  `set_update_kernel_directory(path)` for shared forward shaders before `init`
  or before moving the backend into the rule.

The library compiles its shader source during `init`; a precompiled metallib and
changes to the project CMake files are not required by this implementation.

## Shared Runtime

The general-purpose Metal infrastructure lives in
[`hnnet/metal_runtime.h`](../../../metal_runtime.h), in namespace `hnnet::metal`.
It provides the device context, shared buffers, shader compilation, compute
pipelines, command buffers, dispatch and completion helpers. Its `Context::init`
accepts shader source text and does not depend on backprop kernels or file names.

The local `runtime.h` retains optimizer parameter layouts and a context adapter
that assembles the shared forward shaders with the backprop shader source.
Forward parameter layouts and activation descriptions live beside the update
backend; optimizers keep using the backprop context adapter.

## Reusable Update Backend

[`dense_forward::update::metal::Backend`](../../../nets/dense_forward/update/metal/backend.h)
implements dense propagation independently of backprop. It has two usage paths:

- Standalone: `init(view)`, `to_device(view, states)`, `update()`,
  `to_host(view, states)`. Host states must already contain seeded inputs and
  biases and have both vectors sized to the network. Upload converts weights,
  signals and weighted sums to FP32. `update()` only encodes propagation;
  `to_host()` submits, waits and downloads both state vectors.
- Composed: `init(view, context)` shares the caller's context. Then
  `to_device(view, DeviceData{command, weights, signals, weighted_sums,
  activations, batch_size})` associates existing GPU resources and `update()`
  encodes the forward pass. No host transfers, command submission or waiting
  occur. The caller manages resource lifetime, ordering and completion.

Standalone Objective-C++ command-line callers should enclose the lifecycle in an
`@autoreleasepool`; backprop already scopes its Metal operations this way.

Signal and weighted-sum buffers use sample-major layout with `neuron_count`
stride. Allocate `(batch_size + 1) * neuron_count` floats for each buffer to
accommodate the offsets of MPS matrix views. Weights remain row-major as exposed
by the network; activations use the shared `ActivationDescriptor` layout.
Use resources from the same Metal device as the context.

Backprop uses the composed path immediately after encoding the seed, on the same
command buffer and tensors as loss, backward and optimizer. Forward matrices are
cached until the batch size or associated buffers change. No extra GPU buffer,
transfer or command completion is introduced in this path.

## Compile a Metal Example

Run this from the repository root, set `CLANGXX` to the Objective-C++ capable
LLVM `clang++` you are using, then compile the Metal MNIST example:

```bash
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
mkdir -p build-release/bin
"$CLANGXX" -std=c++26 -O3 -stdlib=libc++ -fobjc-arc -x objective-c++ \
  -isysroot "$SDKROOT" \
  -F"$SDKROOT/System/Library/Frameworks" \
    -Iinclude -Iutils/include \
  examples/backprop/src/backprop_mnist.mm \
    -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -o build-release/bin/hnnet-backprop-mnist-metal
```

The `.mm` source is Objective-C++; the `.cpp` MNIST example remains the OpenMP
version. Run the executable from the repository root so the dataset paths and
default kernel directory recorded from `__FILE__` resolve correctly. No
`-fopenmp` flag is needed for this Metal backend.

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
`init(view)`.

## Supported Operations

- Identity, Sigmoid (including its `sigma` parameter), and ReLU, with derivatives.
- MSE loss (`0.5 * error * error`). Other loss types are not accepted by `backprop`.
- Momentum and iRprop-, with persistent per-weight state on the device.
- `net::DenseForwardNet` only, with contiguous neuron ranges and row-major
  weights. An optional shared bias neuron per layer is supported. Every network
  view argument in the Metal backend is a `DenseForwardNet::View`, so an arbitrary
  `DAGNet` is rejected by the existing backend concept. There is no runtime cast,
  generic DAG or CPU fallback.

The network view exposes read-only `DAGNet::Layer` descriptors through
`layers()`, in topological order. Each descriptor contains transmitter/receiver
ranges and a weight offset. Connected bias neurons are included in the
transmitter range, and the weight row stride equals the transmitter count.
The view also exposes `partition_layer_indices()`: one layer index per partition,
or `DAGNet::Layer::no_layer` for partitions outside a dense layer.
Metal converts these descriptors to its 32-bit kernel format. Input and bias
positions are taken from the corresponding neuron-index spans in the network view.
The common training loop preserves the concrete view type; OpenMP can still
accept it through `DAGNet::View`.

`DenseForwardNet` assembles a `Topology` before constructing its `DAGNet` base.
Neuron and connection construction helpers belong to `Topology`, not to the
constructed network. Dense layers with a single receiver are included in the
descriptors, including single-neuron output layers.

All GPU tensors and arithmetic use FP32. Host `real_t` is unchanged: data and
weights are converted on upload and weights are converted back on download.
Nonfinite inputs, targets, initial weights and activation parameters are rejected.
Results need not be bitwise identical to OpenMP, particularly with host FP64 or
different summation orders. iRprop- may take a different trajectory near zero
gradients because it depends on their signs.

## Execution

`init` reads the network's layer descriptors, checks GPU dimension limits,
compiles the kernels and initializes optimizer buffers. It does not reconstruct
or revalidate the dense topology. There is no adjacency list, topological
scheduler or dense/fallback plan.
MPS operations and matrix views are owned directly by `Backend`; there is no
separate dense backend or wrapper.

`to_device` uploads the dataset and current host weights once per epoch, as in
the existing rule. Input and target buffers are reused and filled directly,
without temporary flattened vectors or a second copy. Data is read again each
epoch so changes to the caller's samples are not silently ignored.

The backend relies on the training protocol: `init(net)` precedes
`to_device`, batch `backprop` calls, and `to_host` for the same network.
Dataset dimensions are validated by `NNet::train`, and nonempty datasets and
batch ranges are handled by `Backprop::learn`; the backend does not repeat
these checks. Direct callers must satisfy the same preconditions. GPU limits,
FP32 conversion checks, and Metal execution errors remain checked.

Forward, hidden-delta propagation and the batch-mean weight gradient use
`MPSMatrixMultiplication`: `X * W^T`, `delta * W`, and `delta^T * X / batch_size`.
Elementwise kernels apply activations and derivatives. Bias contributions and
gradients are included in the same matrix products as other transmitters.
Weights remain in their original order, including bias columns,
for optimizer updates and `to_host`. Matrix views and multiplication objects are
reused for equal batch sizes and rebuilt for a different batch size or new scratch
storage, including a partial final batch.

Each batch still waits for GPU completion before returning its pre-update loss,
because the existing rule requests that scalar synchronously. Removing this wait
would require a separate API decision; it has not been hidden behind stale losses.
`to_host` copies the updated weights back to the network.

Scratch storage grows with the largest batch seen: four FP32 arrays of
`(batch_size + 1) * neuron_count`, plus losses and the connection gradient. The
extra row supplies padding for strided MPS views with a nonzero neuron offset.
Propagated deltas use a separate buffer so matrix multiplication does not need
an in-place operation on overlapping strided matrix views.
Full-batch training can therefore require substantial memory. Tensor indexing is
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
- Dense chains with and without bias, batch size one and a partial final batch.
- Compile-time rejection of a generic `DAGNet` by the Metal backend contract.
- Reset through `init`, upload/download round trips and independent backends.
- Invalid sample shapes, unsupported activations and missing shader files.

Enable Metal API validation and shader validation during these checks. Host-side
descriptor, layout and forwarding checks are in `test/dense_forward_net.cpp`.