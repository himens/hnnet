#pragma once
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>
#include "optimizer_type.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/relu.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense_forward_net.h"

namespace hnnet::backprop::metal {
    template <OptimizerType Optimizer>
        ///////////////////
        // Backend class //
        ///////////////////
        class Backend {
            public:
                // Constructor
                template <typename ...Args>
                    requires std::constructible_from<Optimizer, Args...>
                    explicit Backend(Args&&... args) : _optimizer(std::forward<Args>(args)...) {}
                // Set kernel directory
                void set_kernel_directory(std::filesystem::path directory) {
                    _kernel_directory = std::move(directory);
                }
                // Initialize the backend
                void init(net::DenseForwardNet &net) {
                    @autoreleasepool {
                        _scratch_capacity = 0;
                        _prepared_batch_size = 0;
                        auto view = net.view();
                        const auto neuron_count = detail::checked_count(view.neuron_count());
                        const auto connection_count = detail::checked_count(view.connection_count());
                        if (neuron_count == 0) throw std::invalid_argument("Metal: network must not be empty");
                        const auto layers = view.layers();
                        if (layers.empty()) throw std::invalid_argument("Metal: network must have a dense connection layer");
                        _layers.clear();
                        for (const auto &layer : layers) {
                            if (layer.tx_count <= 0 or layer.rx_count <= 0) {
                                throw std::invalid_argument("Metal: dense layers must be nonempty");
                            }
                            _layers.push_back(Layer{{detail::checked_count(layer.tx_begin), detail::checked_count(layer.tx_count),
                                                    detail::checked_count(layer.rx_begin), detail::checked_count(layer.rx_count),
                                                    detail::checked_count(layer.weight_offset)}});
                        }
                        std::vector<detail::ActivationDescriptor> activations;
                        std::vector<std::uint32_t> input_positions(neuron_count, std::numeric_limits<std::uint32_t>::max());
                        const auto neurons = view.neurons();
                        for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
                            activations.push_back(describe_activation(neurons[neuron].activation()));
                        }
                        const auto &output_layer = _layers.back().shape;
                        const auto input_indices = view.input_neurons_indices();
                        const auto input_count = detail::checked_count(input_indices.size());
                        for (std::uint32_t input = 0; input < input_count; ++input) {
                            input_positions[input_indices[input]] = input;
                        }
                        for (const auto bias_index : view.bias_neurons_indices()) {
                            input_positions[bias_index] = 0xfffffffeu;
                        }
                        _parameters = {neuron_count, connection_count,
                                       input_count, output_layer.rx_count,
                                       output_layer.rx_begin, 0, 0};
                        _context.init(_kernel_directory);
                        _activations = _context.upload(activations);
                        _input_positions = _context.upload(input_positions);
                        _weights = _context.buffer(std::size_t{connection_count} * sizeof(float));
                        _gradient = _context.buffer(std::size_t{connection_count} * sizeof(float));
                        _seed_pipeline = _context.pipeline("hnnet_seed");
                        _loss_pipeline = _context.pipeline("hnnet_loss");
                        _activation_pipeline = _context.pipeline("hnnet_dense_activation");
                        _delta_pipeline = _context.pipeline("hnnet_dense_delta");
                        _optimizer.init(_context, connection_count);
                    }
                }
                // Transfer data to the device
                void to_device(net::DenseForwardNet &net, const DatasetType auto &inputs, const DatasetType auto &targets) {
                    @autoreleasepool {
                        const auto sample_count = detail::checked_count(std::ranges::size(inputs));
                        _inputs = reserve_buffer(_inputs, detail::checked_product(sample_count, _parameters.input_count) * sizeof(float));
                        _targets = reserve_buffer(_targets, detail::checked_product(sample_count, _parameters.output_count) * sizeof(float));
                        auto input_values = static_cast<float *>([_inputs contents]);
                        auto target_values = static_cast<float *>([_targets contents]);
                        std::size_t position = 0;
                        for (const auto &sample : inputs) {
                            for (const auto value : sample) input_values[position++] = to_float(value);
                        }
                        position = 0;
                        for (const auto &sample : targets) {
                            for (const auto value : sample) target_values[position++] = to_float(value);
                        }
                        const auto weights = net.view().weights();
                        auto device_weights = static_cast<float *>([_weights contents]);
                        for (std::size_t edge = 0; edge < weights.size(); ++edge) {
                            device_weights[edge] = to_float(weights[edge]);
                        }
                    }
                }
                // Perform backprop (forward-backward plus weights update)
                real_t backprop(net::DenseForwardNet &, const index_t batch_begin, const index_t batch_end, const hnnet::loss::MSE &) {
                    @autoreleasepool {
                        const auto batch_size = detail::checked_count(batch_end - batch_begin);
                        ensure_scratch(batch_size);
                        auto parameters = _parameters;
                        parameters.batch_begin = static_cast<std::uint32_t>(batch_begin);
                        parameters.batch_size = batch_size;
                                prepare_matrices(parameters);
                        auto command = _context.command();
                        encode(command, _seed_pipeline, parameters,
                               {_activations, _input_positions, _inputs, _signals},
                               detail::checked_product(batch_size, parameters.neuron_count));
                                forward(command, parameters);
                        encode(command, _loss_pipeline, parameters,
                               {_targets, _signals, _losses}, batch_size);
                                backward(command, parameters);
                        _optimizer.update(command, _weights, _gradient, _parameters.connection_count);
                        detail::complete(command);
                        const auto losses = static_cast<const float *>([_losses contents]);
                        real_t loss{0.0_real};
                        for (std::uint32_t sample = 0; sample < batch_size; ++sample) {
                            loss += static_cast<real_t>(losses[sample]);
                        }
                        return loss;
                    }
                }
                // Transfer results back to the host
                void to_host(net::DenseForwardNet &net) {
                    const auto device_weights = static_cast<const float *>([_weights contents]);
                    auto weights = net.view().weights();
                    for (std::size_t edge = 0; edge < weights.size(); ++edge) {
                        weights[edge] = static_cast<real_t>(device_weights[edge]);
                    }
                }
            private:
                // Data types
                struct DenseLayer {
                    std::uint32_t tx_begin;
                    std::uint32_t tx_count;
                    std::uint32_t rx_begin;
                    std::uint32_t rx_count;
                    std::uint32_t weight_offset;
                };
                struct Layer {
                    DenseLayer shape;
                    MPSMatrixMultiplication *forward = nil;
                    MPSMatrixMultiplication *backward = nil;
                    MPSMatrixMultiplication *gradient = nil;
                    MPSMatrix *input = nil;
                    MPSMatrix *weight = nil;
                    MPSMatrix *sum = nil;
                    MPSMatrix *delta = nil;
                    MPSMatrix *previous_delta = nil;
                    MPSMatrix *weight_gradient = nil;
                };
                static_assert(sizeof(DenseLayer) == 20);
                // Helper function to create an MPS matrix view from a buffer
                static MPSMatrix *matrix(id<MTLBuffer> buffer, const std::uint32_t offset,
                                         const std::uint32_t rows, const std::uint32_t columns,
                                         const std::uint32_t stride) {
                    MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                                          columns:columns
                                                                                         rowBytes:std::size_t{stride} * sizeof(float)
                                                                                         dataType:MPSDataTypeFloat32];
                    MPSMatrix *result = [[MPSMatrix alloc] initWithBuffer:buffer
                                                                 offset:std::size_t{offset} * sizeof(float)
                                                             descriptor:descriptor];
                    if (result == nil) throw std::runtime_error("Metal: cannot construct MPS matrix view");
                    return result;
                }

                // Helper function to create an MPS matrix multiplication object
                MPSMatrixMultiplication *product(const bool transpose_left, const bool transpose_right,
                                                 const std::uint32_t rows, const std::uint32_t columns,
                                                 const std::uint32_t interior, const double alpha) const {
                    MPSMatrixMultiplication *result = [[MPSMatrixMultiplication alloc]
                        initWithDevice:_context.device() transposeLeft:transpose_left transposeRight:transpose_right
                        resultRows:rows resultColumns:columns interiorColumns:interior alpha:alpha beta:0.0];
                    if (result == nil) throw std::runtime_error("Metal: cannot construct MPS matrix multiplication");
                    return result;
                }

                void prepare_matrices(const detail::Parameters &parameters) {
                    if (_prepared_batch_size == parameters.batch_size) return;
                    for (auto &layer : _layers) {
                        const auto &shape = layer.shape;
                        layer.forward = product(false, true, parameters.batch_size, shape.rx_count, shape.tx_count, 1.0);
                        layer.backward = product(false, false, parameters.batch_size, shape.tx_count, shape.rx_count, 1.0);
                        layer.gradient = product(true, false, shape.rx_count, shape.tx_count, parameters.batch_size,
                                                 1.0 / static_cast<double>(parameters.batch_size));
                        layer.input = matrix(_signals, shape.tx_begin, parameters.batch_size, shape.tx_count, parameters.neuron_count);
                        layer.weight = matrix(_weights, shape.weight_offset, shape.rx_count, shape.tx_count, shape.tx_count);
                        layer.sum = matrix(_weighted_sums, shape.rx_begin, parameters.batch_size, shape.rx_count, parameters.neuron_count);
                        layer.delta = matrix(_deltas, shape.rx_begin, parameters.batch_size, shape.rx_count, parameters.neuron_count);
                        layer.previous_delta = matrix(_propagated_deltas, shape.tx_begin, parameters.batch_size, shape.tx_count, parameters.neuron_count);
                        layer.weight_gradient = matrix(_gradient, shape.weight_offset, shape.rx_count, shape.tx_count, shape.tx_count);
                    }
                    _prepared_batch_size = parameters.batch_size;
                }
                // Perform the forward pass for the network
                void forward(id<MTLCommandBuffer> command, const detail::Parameters &parameters) const {
                    for (const auto &layer : _layers) {
                        [layer.forward encodeToCommandBuffer:command leftMatrix:layer.input rightMatrix:layer.weight resultMatrix:layer.sum];
                        auto encoder = detail::encoder(command, _activation_pipeline);
                        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
                        [encoder setBytes:&layer.shape length:sizeof(layer.shape) atIndex:1];
                        [encoder setBuffer:_activations offset:0 atIndex:2];
                        [encoder setBuffer:_weighted_sums offset:0 atIndex:3];
                        [encoder setBuffer:_signals offset:0 atIndex:4];
                        detail::dispatch(encoder, _activation_pipeline, detail::checked_product(parameters.batch_size, layer.shape.rx_count));
                    }
                }
                // Perform the backward pass for the network
                void backward(id<MTLCommandBuffer> command, const detail::Parameters &parameters) const {
                    for (std::size_t position = _layers.size(); position > 0; --position) {
                        const auto &layer = _layers[position - 1];
                        auto encoder = detail::encoder(command, _delta_pipeline);
                        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
                        [encoder setBytes:&layer.shape length:sizeof(layer.shape) atIndex:1];
                        [encoder setBuffer:_activations offset:0 atIndex:2];
                        [encoder setBuffer:_targets offset:0 atIndex:3];
                        [encoder setBuffer:_signals offset:0 atIndex:4];
                        [encoder setBuffer:_weighted_sums offset:0 atIndex:5];
                        [encoder setBuffer:_deltas offset:0 atIndex:6];
                        [encoder setBuffer:_propagated_deltas offset:0 atIndex:7];
                        detail::dispatch(encoder, _delta_pipeline, detail::checked_product(parameters.batch_size, layer.shape.rx_count));
                        [layer.gradient encodeToCommandBuffer:command leftMatrix:layer.delta rightMatrix:layer.input resultMatrix:layer.weight_gradient];
                        if (position > 1) {
                            [layer.backward encodeToCommandBuffer:command leftMatrix:layer.delta rightMatrix:layer.weight resultMatrix:layer.previous_delta];
                        }
                    }
                }

                static float to_float(const real_t value) {
                    const auto result = static_cast<float>(value);
                    if (not std::isfinite(result)) {
                        throw std::invalid_argument("Metal: values must be finite and representable in FP32");
                    }
                    return result;
                }
                // Describe the activation function for Metal kernels
                static detail::ActivationDescriptor describe_activation(const Activation *activation) {
                    if (dynamic_cast<const hnnet::activation::Identity *>(activation) != nullptr) return {0, 1.0f};
                    if (const auto sigmoid = dynamic_cast<const hnnet::activation::Sigmoid *>(activation)) {
                        return {1, to_float(sigmoid->sigma)};
                    }
                    if (dynamic_cast<const hnnet::activation::ReLU *>(activation) != nullptr) return {2, 1.0f};
                    throw std::invalid_argument("Metal: supported activations are Identity, Sigmoid and ReLU");
                }
                // Ensure the scratch buffers are large enough for the given batch size
                void ensure_scratch(const std::uint32_t batch_size) {
                    if (batch_size <= _scratch_capacity) return;
                    const auto elements = detail::checked_product(std::size_t{batch_size} + 1, _parameters.neuron_count);
                    _signals = _context.buffer(elements * sizeof(float), false);
                    _weighted_sums = _context.buffer(elements * sizeof(float), false);
                    _deltas = _context.buffer(elements * sizeof(float), false);
                    _propagated_deltas = _context.buffer(elements * sizeof(float), false);
                    _losses = _context.buffer(std::size_t{batch_size} * sizeof(float), false);
                    _scratch_capacity = batch_size;
                    _prepared_batch_size = 0;
                }
                // Reserve a buffer with at least the given number of bytes
                id<MTLBuffer> reserve_buffer(id<MTLBuffer> buffer, const std::size_t bytes) const {
                    if (buffer == nil or [buffer length] < bytes) return _context.buffer(bytes, false);
                    return buffer;
                }
                // Encode a Metal command buffer with the given pipeline and buffers
                static void encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
                                   const detail::Parameters &parameters,
                                   std::initializer_list<id<MTLBuffer>> buffers, const std::size_t count) {
                    auto encoder = detail::encoder(command, pipeline);
                    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
                    NSUInteger index = 1;
                    for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
                    detail::dispatch(encoder, pipeline, count);
                }
            private:
                // Data members
                Optimizer _optimizer;
                detail::Context _context;
                std::filesystem::path _kernel_directory{std::filesystem::path{__FILE__}.parent_path()};
                detail::Parameters _parameters{};
                std::vector<Layer> _layers;
                std::uint32_t _scratch_capacity = 0;
                std::uint32_t _prepared_batch_size = 0;
                id<MTLBuffer> _activations = nil;
                id<MTLBuffer> _input_positions = nil;
                id<MTLBuffer> _inputs = nil;
                id<MTLBuffer> _targets = nil;
                id<MTLBuffer> _weights = nil;
                id<MTLBuffer> _gradient = nil;
                id<MTLBuffer> _signals = nil;
                id<MTLBuffer> _weighted_sums = nil;
                id<MTLBuffer> _deltas = nil;
                id<MTLBuffer> _propagated_deltas = nil;
                id<MTLBuffer> _losses = nil;
                id<MTLComputePipelineState> _seed_pipeline = nil;
                id<MTLComputePipelineState> _loss_pipeline = nil;
                id<MTLComputePipelineState> _activation_pipeline = nil;
                id<MTLComputePipelineState> _delta_pipeline = nil;
        };
}
