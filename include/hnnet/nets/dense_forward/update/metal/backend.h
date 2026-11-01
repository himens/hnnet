#pragma once
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "runtime.h"
#include "hnnet/nets/dense_forward/dense_forward_net.h"

namespace hnnet::dense_forward::update::metal {
    // Resources supplied by a caller that owns GPU execution
    struct DeviceData {
        id<MTLCommandBuffer> command = nil;
        std::uint32_t batch_size = 0;
        id<MTLBuffer> inputs = nil;
        std::uint32_t batch_begin = 0;
    };
    // GPU state buffers exposed without submitting or downloading work
    struct DeviceStates {
        id<MTLBuffer> signals = nil;
        id<MTLBuffer> weighted_sums = nil;
    };

    ///////////////////
    // Backend class //
    ///////////////////
    class Backend {
        public:
            // Set the standalone update shader directory
            void set_kernel_directory(std::filesystem::path directory) {
                _kernel_directory = std::move(directory);
            }
            // Initialize a standalone update context
            void init(net::DenseForwardNet::View view) {
                hnnet::metal::Context context;
                context.init(detail::shader_source(_kernel_directory));
                init(view, context);
            }
            // Share the caller's device, queue and shader library
            void init(net::DenseForwardNet::View view, const hnnet::metal::Context &context) {
                _context = context;
                _parameters = {hnnet::metal::checked_count(view.neuron_count()),
                               hnnet::metal::checked_count(view.connection_count()),
                               hnnet::metal::checked_count(view.input_neurons_indices().size()),
                               hnnet::metal::checked_count(view.output_neurons_indices().size()), 0, 0, 0};
                if (_parameters.neuron_count == 0) throw std::invalid_argument("Metal: network must not be empty");
                const auto layers = view.layers();
                if (layers.empty()) throw std::invalid_argument("Metal: network must have a dense connection layer");
                _layers.clear();
                for (const auto &layer : layers) {
                    if (layer.tx_count <= 0 or layer.rx_count <= 0) {
                        throw std::invalid_argument("Metal: dense layers must be nonempty");
                    }
                    _layers.push_back(Layer{{hnnet::metal::checked_count(layer.tx_begin),
                                            hnnet::metal::checked_count(layer.tx_count),
                                            hnnet::metal::checked_count(layer.rx_begin),
                                            hnnet::metal::checked_count(layer.rx_count),
                                            hnnet::metal::checked_count(layer.weight_offset)}});
                }
                _parameters.output_begin = _layers.back().shape.rx_begin;
                std::vector<std::uint32_t> input_positions(_parameters.neuron_count, std::numeric_limits<std::uint32_t>::max());
                const auto input_indices = view.input_neurons_indices();
                for (std::uint32_t input = 0; input < _parameters.input_count; ++input) {
                    input_positions[input_indices[input]] = input;
                }
                for (const auto bias_index : view.bias_neurons_indices()) {
                    input_positions[bias_index] = 0xfffffffeu;
                }
                _input_positions = _context.upload(input_positions);
                std::vector<detail::ActivationDescriptor> activations;
                for (const auto &neuron : view.neurons()) {
                    activations.push_back(detail::describe_activation(neuron.activation()));
                }
                _owned_activations = _context.upload(activations);
                _seed_pipeline = _context.pipeline("hnnet_seed");
                _activation_pipeline = _context.pipeline("hnnet_dense_activation");
                _data = {};
                _prepared_batch_size = 0;
                _state_capacity = 0;
                _host_batch = false;
                _encoded = false;
                _device_weights = nil;
                _owned_weights = nil;
                _owned_signals = nil;
                _owned_weighted_sums = nil;
                _owned_inputs = nil;
            }
            // Associate existing device resources without uploading or submitting work
            void to_device(const DeviceData &data) {
                if (_activation_pipeline == nil) {
                    throw std::logic_error("Metal: device association requires init");
                }
                if (data.command == nil or data.inputs == nil) {
                    throw std::invalid_argument("Metal: update requires a command and input buffer");
                }
                if ([data.command device] != _context.device() or
                    [data.inputs device] != _context.device()) {
                    throw std::invalid_argument("Metal: update resources must use the context device");
                }
                const auto sample_end = std::size_t{data.batch_begin} + data.batch_size;
                hnnet::metal::checked_count(sample_end);
                const auto input_elements = hnnet::metal::checked_product(sample_end, _parameters.input_count);
                if ([data.inputs length] < input_elements * sizeof(float)) {
                    throw std::invalid_argument("Metal: update buffers are smaller than the batch");
                }
                ensure_states(data.batch_size);
                _data = data;
                _parameters.batch_size = data.batch_size;
                _parameters.batch_begin = data.batch_begin;
                _host_batch = false;
                _encoded = false;
            }
            // Upload a dataset for standalone inference
            void to_device(const DatasetType auto &inputs) {
                if (_activation_pipeline == nil) {
                    throw std::logic_error("Metal: host upload requires init");
                }
                const auto batch_size = hnnet::metal::checked_count(std::ranges::size(inputs));
                for (const auto &input : inputs) {
                    if (std::ranges::size(input) != _parameters.input_count) {
                        throw std::invalid_argument("Metal: input size does not match the network");
                    }
                }
                const auto input_elements = hnnet::metal::checked_product(batch_size, _parameters.input_count);
                _owned_inputs = reserve_buffer(_owned_inputs, input_elements * sizeof(float));
                auto input_values = static_cast<float *>([_owned_inputs contents]);
                std::size_t offset = 0;
                for (const auto &input : inputs) {
                    for (const auto value : input) {
                        input_values[offset++] = detail::to_float(value);
                    }
                }
                to_device(DeviceData{_context.command(), batch_size, _owned_inputs, 0});
                _host_batch = true;
            }
            // Upload host weights once for standalone inference
            void update_states(const DataType auto &weights) {
                if (not _host_batch or _encoded) {
                    throw std::logic_error("Metal: host update requires a fresh standalone upload");
                }
                if (std::ranges::size(weights) != _parameters.connection_count) {
                    throw std::invalid_argument("Metal: weight counts must match");
                }
                _owned_weights = reserve_buffer(_owned_weights, std::size_t{_parameters.connection_count} * sizeof(float));
                auto device_weights = static_cast<float *>([_owned_weights contents]);
                for (const auto &[index, weight] : weights | std::views::enumerate) {
                    device_weights[index] = detail::to_float(weight);
                }
                update_states(_owned_weights);
            }
            // Encode seed and propagation without submitting or waiting for the command
            void update_states(id<MTLBuffer> weights) {
                if (_activation_pipeline == nil or _data.command == nil) {
                    throw std::logic_error("Metal: update requires init and to_device");
                }
                if (_encoded) {
                    throw std::logic_error("Metal: update requires a fresh upload or device association");
                }
                if (weights == nil or [weights device] != _context.device() or
                    [weights length] < std::size_t{_parameters.connection_count} * sizeof(float)) {
                    throw std::invalid_argument("Metal: update weights must match the context and network");
                }
                if (weights != _device_weights) {
                    _prepared_batch_size = 0;
                }
                _device_weights = weights;
                _encoded = true;
                if (_parameters.batch_size == 0) return;
                prepare_matrices();
                auto seed_encoder = hnnet::metal::encoder(_data.command, _seed_pipeline);
                [seed_encoder setBytes:&_parameters length:sizeof(_parameters) atIndex:0];
                [seed_encoder setBuffer:_owned_activations offset:0 atIndex:1];
                [seed_encoder setBuffer:_input_positions offset:0 atIndex:2];
                [seed_encoder setBuffer:_data.inputs offset:0 atIndex:3];
                [seed_encoder setBuffer:_owned_signals offset:0 atIndex:4];
                [seed_encoder setBuffer:_owned_weighted_sums offset:0 atIndex:5];
                hnnet::metal::dispatch(seed_encoder, _seed_pipeline,
                                      hnnet::metal::checked_product(_parameters.batch_size, _parameters.neuron_count));
                for (const auto &layer : _layers) {
                    [layer.forward encodeToCommandBuffer:_data.command leftMatrix:layer.input rightMatrix:layer.weight resultMatrix:layer.sum];
                    auto encoder = hnnet::metal::encoder(_data.command, _activation_pipeline);
                    [encoder setBytes:&_parameters length:sizeof(_parameters) atIndex:0];
                    [encoder setBytes:&layer.shape length:sizeof(layer.shape) atIndex:1];
                    [encoder setBuffer:_owned_activations offset:0 atIndex:2];
                    [encoder setBuffer:_owned_weighted_sums offset:0 atIndex:3];
                    [encoder setBuffer:_owned_signals offset:0 atIndex:4];
                    hnnet::metal::dispatch(encoder, _activation_pipeline,
                                          hnnet::metal::checked_product(_parameters.batch_size, layer.shape.rx_count));
                }
            }
            // Expose GPU states to the caller without copying or waiting
            void to_host(DeviceStates &states) const {
                if (not _encoded) {
                    throw std::logic_error("Metal: state access requires encoded update");
                }
                states = {_owned_signals, _owned_weighted_sums};
            }
            // Complete standalone work and download the output dataset
            void to_host(dataset_vector_t &outputs) {
                if (not _host_batch or not _encoded) {
                    throw std::logic_error("Metal: output download requires a standalone update");
                }
                hnnet::metal::complete(_data.command);
                const auto signals = static_cast<const float *>([_owned_signals contents]);
                outputs.resize(_parameters.batch_size);
                for (std::size_t sample = 0; sample < outputs.size(); ++sample) {
                    auto &output = outputs[sample];
                    output.resize(_parameters.output_count);
                    const auto offset = sample * _parameters.neuron_count + _parameters.output_begin;
                    for (std::size_t neuron = 0; neuron < output.size(); ++neuron) {
                        output[neuron] = static_cast<real_t>(signals[offset + neuron]);
                    }
                }
                _data.command = nil;
                _host_batch = false;
                _encoded = false;
            }
        private:
            // Data types
            struct Layer {
                detail::DenseLayer shape;
                MPSMatrixMultiplication *forward = nil;
                MPSMatrix *input = nil;
                MPSMatrix *weight = nil;
                MPSMatrix *sum = nil;
            };
            // Cache forward matrices until batch dimensions or buffers change
            void prepare_matrices() {
                if (_prepared_batch_size == _parameters.batch_size) return;
                for (auto &layer : _layers) {
                    const auto &shape = layer.shape;
                    layer.forward = detail::product(_context.device(), false, true, _parameters.batch_size, shape.rx_count, shape.tx_count, 1.0);
                    layer.input = detail::matrix(_owned_signals, shape.tx_begin, _parameters.batch_size, shape.tx_count, _parameters.neuron_count);
                    layer.weight = detail::matrix(_device_weights, shape.weight_offset, shape.rx_count, shape.tx_count, shape.tx_count);
                    layer.sum = detail::matrix(_owned_weighted_sums, shape.rx_begin, _parameters.batch_size, shape.rx_count, _parameters.neuron_count);
                }
                _prepared_batch_size = _parameters.batch_size;
            }
            // Reuse batch states; MPS views require an extra row for offsets
            void ensure_states(const std::uint32_t batch_size) {
                if (batch_size <= _state_capacity and _owned_signals != nil) return;
                const auto elements = hnnet::metal::checked_product(std::size_t{batch_size} + 1, _parameters.neuron_count);
                _owned_signals = _context.buffer(elements * sizeof(float), false);
                _owned_weighted_sums = _context.buffer(elements * sizeof(float), false);
                _state_capacity = batch_size;
                _prepared_batch_size = 0;
            }
            // Grow an owned buffer only when necessary
            id<MTLBuffer> reserve_buffer(id<MTLBuffer> buffer, const std::size_t bytes) const {
                if (buffer == nil or [buffer length] < bytes) return _context.buffer(bytes, false);
                return buffer;
            }
        private:
            // Data members
            hnnet::metal::Context _context;
            std::filesystem::path _kernel_directory{detail::kernel_directory()};
            detail::Parameters _parameters{};
            DeviceData _data{};
            std::vector<Layer> _layers;
            std::uint32_t _prepared_batch_size = 0;
            std::uint32_t _state_capacity = 0;
            bool _host_batch = false;
            bool _encoded = false;
            id<MTLBuffer> _device_weights = nil;
            id<MTLComputePipelineState> _activation_pipeline = nil;
            id<MTLComputePipelineState> _seed_pipeline = nil;
            id<MTLBuffer> _input_positions = nil;
            id<MTLBuffer> _owned_activations = nil;
            id<MTLBuffer> _owned_weights = nil;
            id<MTLBuffer> _owned_signals = nil;
            id<MTLBuffer> _owned_weighted_sums = nil;
            id<MTLBuffer> _owned_inputs = nil;
    };
}
