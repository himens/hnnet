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
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>
#include "optimizer_type.h"
#include "hnnet/activations/identity.h"
#include "hnnet/activations/relu.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dag_net.h"

namespace hnnet::backprop::metal {
    template <OptimizerType Optimizer>
        class Backend {
            private:
                struct Level {
                    std::uint32_t begin;
                    std::uint32_t count;
                };
            public:
                template <typename ...Args>
                    requires std::constructible_from<Optimizer, Args...>
                    explicit Backend(Args&&... args) : _optimizer(std::forward<Args>(args)...) {}

                void set_kernel_directory(std::filesystem::path directory) {
                    if (_ready) throw std::logic_error("Metal: set kernel directory before init");
                    _kernel_directory = std::move(directory);
                }

                void init(net::DAGNet &net) {
                    @autoreleasepool {
                        _ready = false;
                        _uploaded = false;
                        _scratch_capacity = 0;
                        _net = nullptr;
                        auto view = net.view();
                        const auto neuron_count = detail::checked_count(view.neuron_count());
                        const auto connection_count = detail::checked_count(view.connection_count());
                        if (neuron_count == 0) throw std::invalid_argument("Metal: network must not be empty");
                        detail::checked_count(std::size_t{neuron_count} + 1);

                        std::vector<detail::ActivationDescriptor> activations;
                        std::vector<std::uint32_t> input_neurons, bias_neurons;
                        std::vector<std::uint32_t> output_neurons;
                        std::vector<std::uint32_t> output_positions(neuron_count, std::numeric_limits<std::uint32_t>::max());
                        const auto neurons = view.neurons();
                        for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
                            activations.push_back(describe_activation(neurons[neuron].activation()));
                            if (neurons[neuron].type() == NeuronType::input) input_neurons.push_back(neuron);
                            if (neurons[neuron].type() == NeuronType::bias) bias_neurons.push_back(neuron);
                        }
                        for (const auto neuron : view.iout_neurons()) {
                            if (neuron < 0 or neuron >= neuron_count) {
                                throw std::invalid_argument("Metal: invalid output neuron");
                            }
                            output_positions[neuron] = detail::checked_count(output_neurons.size());
                            output_neurons.push_back(static_cast<std::uint32_t>(neuron));
                        }
                        if (input_neurons.empty() or output_neurons.empty()) {
                            throw std::invalid_argument("Metal: network needs input and output neurons");
                        }

                        std::vector<std::uint32_t> sources, receivers;
                        std::vector<std::uint32_t> incoming_offsets(std::size_t{neuron_count} + 1, 0);
                        std::vector<std::uint32_t> outgoing_offsets(std::size_t{neuron_count} + 1, 0);
                        for (const auto &connection : view.connections()) {
                            if (connection.itx < 0 or connection.itx >= neuron_count or
                                connection.irx < 0 or connection.irx >= neuron_count) {
                                throw std::invalid_argument("Metal: invalid connection");
                            }
                            if (neurons[connection.irx].type() == NeuronType::input or
                                neurons[connection.irx].type() == NeuronType::bias) {
                                throw std::invalid_argument("Metal: input and bias neurons cannot receive connections");
                            }
                            sources.push_back(static_cast<std::uint32_t>(connection.itx));
                            receivers.push_back(static_cast<std::uint32_t>(connection.irx));
                            ++incoming_offsets[connection.irx + 1];
                            ++outgoing_offsets[connection.itx + 1];
                        }
                        std::partial_sum(incoming_offsets.begin(), incoming_offsets.end(), incoming_offsets.begin());
                        std::partial_sum(outgoing_offsets.begin(), outgoing_offsets.end(), outgoing_offsets.begin());
                        std::vector<std::uint32_t> incoming_edges(connection_count), outgoing_edges(connection_count);
                        auto incoming_cursor = incoming_offsets;
                        auto outgoing_cursor = outgoing_offsets;
                        for (std::uint32_t edge = 0; edge < connection_count; ++edge) {
                            incoming_edges[incoming_cursor[receivers[edge]]++] = edge;
                            outgoing_edges[outgoing_cursor[sources[edge]]++] = edge;
                        }

                        std::vector<std::uint32_t> degree(neuron_count), depth(neuron_count, 0), queue;
                        for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
                            degree[neuron] = incoming_offsets[neuron + 1] - incoming_offsets[neuron];
                            if (degree[neuron] == 0) queue.push_back(neuron);
                        }
                        for (std::size_t position = 0; position < queue.size(); ++position) {
                            const auto neuron = queue[position];
                            for (auto entry = outgoing_offsets[neuron]; entry < outgoing_offsets[neuron + 1]; ++entry) {
                                const auto receiver = receivers[outgoing_edges[entry]];
                                depth[receiver] = std::max(depth[receiver], depth[neuron] + 1);
                                if (--degree[receiver] == 0) queue.push_back(receiver);
                            }
                        }
                        if (queue.size() != neuron_count) throw std::invalid_argument("Metal: network contains a cycle");
                        std::vector<std::vector<std::uint32_t>> grouped(std::size_t{std::ranges::max(depth)} + 1);
                        for (const auto neuron : queue) {
                            if (incoming_offsets[neuron] != incoming_offsets[neuron + 1]) {
                                grouped[depth[neuron]].push_back(neuron);
                            }
                        }
                        std::vector<std::uint32_t> level_neurons;
                        _levels.clear();
                        for (const auto &group : grouped) {
                            if (group.empty()) continue;
                            _levels.push_back({detail::checked_count(level_neurons.size()), detail::checked_count(group.size())});
                            level_neurons.insert(level_neurons.end(), group.begin(), group.end());
                        }

                        _parameters = {neuron_count, connection_count,
                                       detail::checked_count(input_neurons.size()),
                                       detail::checked_count(output_neurons.size()),
                                       detail::checked_count(bias_neurons.size()), 0, 0, 0, 0};
                        _context.init(_kernel_directory);
                        _activations = _context.upload(activations);
                        _input_neurons = _context.upload(input_neurons);
                        _bias_neurons = _context.upload(bias_neurons);
                        _output_neurons = _context.upload(output_neurons);
                        _output_positions = _context.upload(output_positions);
                        _sources = _context.upload(sources);
                        _receivers = _context.upload(receivers);
                        _incoming_offsets = _context.upload(incoming_offsets);
                        _outgoing_offsets = _context.upload(outgoing_offsets);
                        _incoming_edges = _context.upload(incoming_edges);
                        _outgoing_edges = _context.upload(outgoing_edges);
                        _level_neurons = _context.upload(level_neurons);
                        _weights = _context.buffer(std::size_t{connection_count} * sizeof(float));
                        _gradient = _context.buffer(std::size_t{connection_count} * sizeof(float));
                        _seed_pipeline = _context.pipeline("hnnet_seed");
                        _forward_pipeline = _context.pipeline("hnnet_forward");
                        _backward_pipeline = _context.pipeline("hnnet_backward");
                        _loss_pipeline = _context.pipeline("hnnet_loss");
                        _gradient_pipeline = _context.pipeline("hnnet_gradient");
                        _optimizer.init(_context, connection_count);
                        _net = &net;
                        _ready = true;
                    }
                }

                void to_device(net::DAGNet &net, const DatasetType auto &inputs, const DatasetType auto &targets) {
                    @autoreleasepool {
                        require_ready(net);
                        _uploaded = false;
                        const auto sample_count = detail::checked_count(std::ranges::size(inputs));
                        if (sample_count == 0 or std::ranges::size(targets) != sample_count) {
                            throw std::invalid_argument("Metal: input and target sample counts must match and be nonzero");
                        }
                        std::vector<float> input_values, target_values;
                        input_values.reserve(detail::checked_product(sample_count, _parameters.input_count));
                        target_values.reserve(detail::checked_product(sample_count, _parameters.output_count));
                        for (const auto &sample : inputs) {
                            if (std::ranges::size(sample) != _parameters.input_count) {
                                throw std::invalid_argument("Metal: input sample size mismatch");
                            }
                            for (const auto value : sample) input_values.push_back(to_float(value));
                        }
                        for (const auto &sample : targets) {
                            if (std::ranges::size(sample) != _parameters.output_count) {
                                throw std::invalid_argument("Metal: target sample size mismatch");
                            }
                            for (const auto value : sample) target_values.push_back(to_float(value));
                        }
                        _inputs = _context.upload(input_values);
                        _targets = _context.upload(target_values);
                        const auto weights = net.view().weights();
                        auto device_weights = static_cast<float *>([_weights contents]);
                        for (std::size_t edge = 0; edge < weights.size(); ++edge) {
                            device_weights[edge] = to_float(weights[edge]);
                        }
                        _sample_count = sample_count;
                        _uploaded = true;
                    }
                }

                real_t backprop(net::DAGNet &net, const index_t batch_begin, const index_t batch_end,
                                const hnnet::loss::MSE &) {
                    @autoreleasepool {
                        require_ready(net);
                        if (not _uploaded or batch_begin < 0 or batch_end <= batch_begin or batch_end > _sample_count) {
                            throw std::invalid_argument("Metal: invalid batch range or missing to_device");
                        }
                        const auto batch_size = detail::checked_count(batch_end - batch_begin);
                        ensure_scratch(batch_size);
                        auto parameters = _parameters;
                        parameters.batch_begin = static_cast<std::uint32_t>(batch_begin);
                        parameters.batch_size = batch_size;
                        auto command = _context.command();
                        encode(command, _seed_pipeline, parameters,
                               {_activations, _input_neurons, _bias_neurons, _inputs, _signals, _weighted_sums, _deltas},
                               batch_size);
                        for (const auto &level : _levels) {
                            parameters.level_begin = level.begin;
                            parameters.level_count = level.count;
                            encode(command, _forward_pipeline, parameters,
                                   {_activations, _level_neurons, _incoming_offsets, _incoming_edges,
                                    _sources, _weights, _signals, _weighted_sums},
                                   detail::checked_product(batch_size, level.count));
                        }
                        encode(command, _loss_pipeline, parameters,
                               {_output_neurons, _targets, _signals, _losses}, batch_size);
                        for (const auto &level : _levels | std::views::reverse) {
                            parameters.level_begin = level.begin;
                            parameters.level_count = level.count;
                            encode(command, _backward_pipeline, parameters,
                                   {_activations, _level_neurons, _outgoing_offsets, _outgoing_edges, _receivers,
                                    _weights, _output_positions, _targets, _signals, _weighted_sums, _deltas},
                                   detail::checked_product(batch_size, level.count));
                        }
                        encode(command, _gradient_pipeline, parameters,
                               {_sources, _receivers, _signals, _deltas, _gradient}, _parameters.connection_count);
                        _optimizer.update(command, _weights, _gradient, _parameters.connection_count);
                        try {
                            detail::complete(command);
                        }
                        catch (...) {
                            _uploaded = false;
                            throw;
                        }
                        const auto losses = static_cast<const float *>([_losses contents]);
                        real_t loss{0.0_real};
                        for (std::uint32_t sample = 0; sample < batch_size; ++sample) {
                            loss += static_cast<real_t>(losses[sample]);
                        }
                        return loss;
                    }
                }

                void to_host(net::DAGNet &net) {
                    require_ready(net);
                    if (not _uploaded) throw std::logic_error("Metal: missing or failed device computation");
                    const auto device_weights = static_cast<const float *>([_weights contents]);
                    auto weights = net.view().weights();
                    for (std::size_t edge = 0; edge < weights.size(); ++edge) {
                        weights[edge] = static_cast<real_t>(device_weights[edge]);
                    }
                }

            private:
                static float to_float(const real_t value) {
                    const auto result = static_cast<float>(value);
                    if (not std::isfinite(result)) {
                        throw std::invalid_argument("Metal: values must be finite and representable in FP32");
                    }
                    return result;
                }

                static detail::ActivationDescriptor describe_activation(const Activation *activation) {
                    if (dynamic_cast<const hnnet::activation::Identity *>(activation) != nullptr) return {0, 1.0f};
                    if (const auto sigmoid = dynamic_cast<const hnnet::activation::Sigmoid *>(activation)) {
                        return {1, to_float(sigmoid->sigma)};
                    }
                    if (dynamic_cast<const hnnet::activation::ReLU *>(activation) != nullptr) return {2, 1.0f};
                    throw std::invalid_argument("Metal: supported activations are Identity, Sigmoid and ReLU");
                }

                void require_ready(net::DAGNet &net) const {
                    if (not _ready or _net != &net) throw std::logic_error("Metal: call init with this network first");
                    const auto view = net.view();
                    if (view.neuron_count() != _parameters.neuron_count or
                        view.connection_count() != _parameters.connection_count) {
                        throw std::logic_error("Metal: network topology changed; call init again");
                    }
                }

                void ensure_scratch(const std::uint32_t batch_size) {
                    if (batch_size <= _scratch_capacity) return;
                    const auto elements = detail::checked_product(batch_size, _parameters.neuron_count);
                    _signals = _context.buffer(elements * sizeof(float));
                    _weighted_sums = _context.buffer(elements * sizeof(float));
                    _deltas = _context.buffer(elements * sizeof(float));
                    _losses = _context.buffer(std::size_t{batch_size} * sizeof(float));
                    _scratch_capacity = batch_size;
                }

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
                Optimizer _optimizer;
                detail::Context _context;
                std::filesystem::path _kernel_directory{std::filesystem::path{__FILE__}.parent_path()};
                detail::Parameters _parameters{};
                std::vector<Level> _levels;
                net::DAGNet *_net = nullptr;
                std::uint32_t _sample_count = 0;
                std::uint32_t _scratch_capacity = 0;
                bool _ready = false;
                bool _uploaded = false;
                id<MTLBuffer> _activations = nil;
                id<MTLBuffer> _input_neurons = nil;
                id<MTLBuffer> _bias_neurons = nil;
                id<MTLBuffer> _output_neurons = nil;
                id<MTLBuffer> _output_positions = nil;
                id<MTLBuffer> _sources = nil;
                id<MTLBuffer> _receivers = nil;
                id<MTLBuffer> _incoming_offsets = nil;
                id<MTLBuffer> _outgoing_offsets = nil;
                id<MTLBuffer> _incoming_edges = nil;
                id<MTLBuffer> _outgoing_edges = nil;
                id<MTLBuffer> _level_neurons = nil;
                id<MTLBuffer> _inputs = nil;
                id<MTLBuffer> _targets = nil;
                id<MTLBuffer> _weights = nil;
                id<MTLBuffer> _gradient = nil;
                id<MTLBuffer> _signals = nil;
                id<MTLBuffer> _weighted_sums = nil;
                id<MTLBuffer> _deltas = nil;
                id<MTLBuffer> _losses = nil;
                id<MTLComputePipelineState> _seed_pipeline = nil;
                id<MTLComputePipelineState> _forward_pipeline = nil;
                id<MTLComputePipelineState> _backward_pipeline = nil;
                id<MTLComputePipelineState> _loss_pipeline = nil;
                id<MTLComputePipelineState> _gradient_pipeline = nil;
        };
}