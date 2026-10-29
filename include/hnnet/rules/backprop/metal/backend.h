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
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include "optimizer_type.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/relu.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense_forward/dense_forward_net.h"
#include "hnnet/nets/dense_forward/update/metal/backend.h"
#include "hnnet/rules/backprop/backend_type.h"

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
                // Set the shared forward shader directory
                void set_update_kernel_directory(std::filesystem::path directory) {
                    _update_kernel_directory = std::move(directory);
                }
                // Initialize the backend
                void init(net::DenseForwardNet::View &view) {
                    @autoreleasepool {
                        _scratch_capacity = 0;
                        _prepared_batch_size = 0;
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
                        const auto neurons = view.neurons();
                        for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
                            activations.push_back(detail::describe_activation(neurons[neuron].activation()));
                        }
                        const auto &output_layer = _layers.back().shape;
                        const auto input_indices = view.input_neurons_indices();
                        const auto input_count = detail::checked_count(input_indices.size());
                        _parameters = {neuron_count, connection_count,
                                       input_count, output_layer.rx_count,
                                       output_layer.rx_begin, 0, 0};
                        _context.init(_kernel_directory, _update_kernel_directory);
                        _update_backend.init(view, _context);
                        _activations = _context.upload(activations);
                        _weights = _context.buffer(std::size_t{connection_count} * sizeof(float));
                        _gradient = _context.buffer(std::size_t{connection_count} * sizeof(float));
                        _loss_pipeline = _context.pipeline("hnnet_loss");
                        _delta_pipeline = _context.pipeline("hnnet_dense_delta");
                        _optimizer.init(_context, connection_count);
                    }
                }
                // Transfer data to the device
                template <DatasetType Inputs, DatasetType Targets>
                    void to_device(net::DenseForwardNet::View &view, const backprop::TrainingData<Inputs, Targets> &data) {
                        @autoreleasepool {
                            const auto sample_count = detail::checked_count(std::ranges::size(data.inputs));
                            _inputs = reserve_buffer(_inputs, detail::checked_product(sample_count, _parameters.input_count) * sizeof(float));
                            _targets = reserve_buffer(_targets, detail::checked_product(sample_count, _parameters.output_count) * sizeof(float));
                            auto input_values = static_cast<float *>([_inputs contents]);
                            auto target_values = static_cast<float *>([_targets contents]);
                            std::size_t position = 0;
                            for (const auto &sample : data.inputs) {
                                for (const auto value : sample) input_values[position++] = detail::to_float(value);
                            }
                            position = 0;
                            for (const auto &sample : data.targets) {
                                for (const auto value : sample) target_values[position++] = detail::to_float(value);
                            }
                            const auto weights = view.weights();
                            auto device_weights = static_cast<float *>([_weights contents]);
                            for (std::size_t edge = 0; edge < weights.size(); ++edge) {
                                device_weights[edge] = detail::to_float(weights[edge]);
                            }
                        }
                    }
                // Perform backprop (forward-backward plus weights update)
                real_t backprop(net::DenseForwardNet::View &view,
                                const index_t batch_begin,
                                const index_t batch_end,
                                const loss::MSE &) {
                    @autoreleasepool {
                        const auto batch_size = detail::checked_count(batch_end - batch_begin);
                        ensure_scratch(batch_size);
                        auto parameters = _parameters;
                        parameters.batch_begin = static_cast<std::uint32_t>(batch_begin);
                        parameters.batch_size = batch_size;
                        prepare_matrices(parameters);
                        auto command = _context.command();
                        _update_backend.to_device(view, dense_forward::update::metal::DeviceData{
                            command, _weights, _signals, _weighted_sums, _activations, batch_size,
                            _inputs, parameters.batch_begin});
                        _update_backend.update();
                        encode(command, _loss_pipeline, parameters, {_targets, _signals, _losses}, batch_size);
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
                void to_host(net::DenseForwardNet::View &, std::span<real_t> weights) {
                    const auto device_weights = static_cast<const float *>([_weights contents]);
                    for (std::size_t edge = 0; edge < weights.size(); ++edge) {
                        weights[edge] = static_cast<real_t>(device_weights[edge]);
                    }
                }
            private:
                // Data types
                using DenseLayer = detail::DenseLayer;
                struct Layer {
                    DenseLayer shape;
                    MPSMatrixMultiplication *backward = nil;
                    MPSMatrixMultiplication *gradient = nil;
                    MPSMatrix *input = nil;
                    MPSMatrix *weight = nil;
                    MPSMatrix *delta = nil;
                    MPSMatrix *previous_delta = nil;
                    MPSMatrix *weight_gradient = nil;
                };
                static_assert(sizeof(DenseLayer) == 20);
                // Prepare the matrices used by backward and gradient computation
                void prepare_matrices(const detail::Parameters &parameters) {
                    if (_prepared_batch_size == parameters.batch_size) return;
                    for (auto &layer : _layers) {
                        const auto &shape = layer.shape;
                        layer.backward = detail::product(_context.device(), false, false, parameters.batch_size, shape.tx_count, shape.rx_count, 1.0);
                        layer.gradient = detail::product(_context.device(), true, false, shape.rx_count, shape.tx_count, parameters.batch_size,
                                                 1.0 / static_cast<double>(parameters.batch_size));
                        layer.input = detail::matrix(_signals, shape.tx_begin, parameters.batch_size, shape.tx_count, parameters.neuron_count);
                        layer.weight = detail::matrix(_weights, shape.weight_offset, shape.rx_count, shape.tx_count, shape.tx_count);
                        layer.delta = detail::matrix(_deltas, shape.rx_begin, parameters.batch_size, shape.rx_count, parameters.neuron_count);
                        layer.previous_delta = detail::matrix(_propagated_deltas, shape.tx_begin, parameters.batch_size, shape.tx_count, parameters.neuron_count);
                        layer.weight_gradient = detail::matrix(_gradient, shape.weight_offset, shape.rx_count, shape.tx_count, shape.tx_count);
                    }
                    _prepared_batch_size = parameters.batch_size;
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
                dense_forward::update::metal::Backend _update_backend;
                std::filesystem::path _kernel_directory{std::filesystem::path{__FILE__}.parent_path()};
                std::filesystem::path _update_kernel_directory{dense_forward::update::metal::detail::kernel_directory()};
                detail::Parameters _parameters{};
                std::vector<Layer> _layers;
                std::uint32_t _scratch_capacity = 0;
                std::uint32_t _prepared_batch_size = 0;
                id<MTLBuffer> _activations = nil;
                id<MTLBuffer> _inputs = nil;
                id<MTLBuffer> _targets = nil;
                id<MTLBuffer> _weights = nil;
                id<MTLBuffer> _gradient = nil;
                id<MTLBuffer> _signals = nil;
                id<MTLBuffer> _weighted_sums = nil;
                id<MTLBuffer> _deltas = nil;
                id<MTLBuffer> _propagated_deltas = nil;
                id<MTLBuffer> _losses = nil;
                id<MTLComputePipelineState> _loss_pipeline = nil;
                id<MTLComputePipelineState> _delta_pipeline = nil;
        };
}
