#pragma once
#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>
#include <omp.h>
#include "optimizer_type.h"
#include "hnnet/rules/backprop/backend_type.h"
#include "hnnet/loss_type.h"
#include "hnnet/nets/dag/dag_net.h"
#include "hnnet/nets/dag/update/omp/backend.h"

namespace hnnet::backprop::omp {
    ///////////////////
    // Backend class //
    ///////////////////
    template <OptimizerType Optimizer>
        class Backend {
            public:
                // Constructor
                template <typename ...Args>
                    requires std::constructible_from<Optimizer, Args...>
                    explicit Backend(Args&&... args) : _optimizer(std::forward<Args>(args)...) {}
                // Initialize the backend
                void init(net::DAGNet::View &view) {
                    const auto max_threads = omp_get_max_threads();
                    const auto neuron_count = view.neuron_count();
                    const auto connection_count = view.connection_count();
                    const auto weights = view.weights();
                    _weights.assign(weights.begin(), weights.end());            
                    _deltas.assign(max_threads, std::vector<real_t>(neuron_count, 0.0_real));
                    _gradients.assign(max_threads, std::vector<real_t>(connection_count, 0.0_real));
                    _batch_gradient.assign(connection_count, 0.0_real);
                    _optimizer.init(connection_count);
                    _targets.clear();
                    _view = view;
                    _update_backend.init(view);
                }
                // Transfer data to the device
                void to_device(const TrainingDataType auto &data) {
                    _targets.clear();
                    for (const auto &target : data.targets) {
                        _targets.emplace_back(target);
                    }
                    _update_backend.to_device(data.inputs);
                }
                // Transfer data back to the host
                void to_host(std::vector<real_t> &weights) {
                    weights = std::move(_weights);
                }
                // Perform backprop (forward-backward plus weights update)
                real_t backprop(const index_t batch_begin, const index_t batch_end, const LossType auto &loss) {
                    if (batch_begin < 0 or batch_end <= batch_begin or batch_end > std::ssize(_targets)) {
                        throw std::invalid_argument("Backend::backprop: batch range out of bounds");
                    }
                    std::span<const NNet::State> states;
                    _update_backend.update_states(_weights, batch_begin, batch_end);
                    _update_backend.to_host(states);
                    const auto batch_size = batch_end - batch_begin;
                    const auto parallel = batch_size > 1;
                    const auto thread_count = parallel ? std::ssize(_gradients) : index_t{1};
                    for (auto thread_index{0}; thread_index < thread_count; ++thread_index) {
                        std::ranges::fill(_gradients[thread_index], 0.0_real);
                    }
                    real_t loss_value{0.0_real};
                    #pragma omp parallel for if(parallel) num_threads(thread_count) reduction(+:loss_value)
                    for (auto input_index = batch_begin; input_index < batch_end; ++input_index) {
                        const auto thread_index = omp_get_thread_num();
                        loss_value += backward(thread_index, input_index, states[input_index - batch_begin], loss);
                    }
                    if (parallel) {
                        std::ranges::fill(_batch_gradient, 0.0_real);
                        for (auto thread_index{0}; thread_index < thread_count; ++thread_index) {
                            const auto &gradient = _gradients[thread_index];
                            for (auto connection_index{0}; connection_index < _view.connection_count(); ++connection_index) {
                                _batch_gradient[connection_index] += gradient[connection_index] / batch_size;
                            }
                        }
                    }
                    _optimizer.update_weights(_weights, parallel ? _batch_gradient : _gradients.front());
                    return loss_value;
                }
            protected:
                // Compute the error and delta weights contribution of a single sample
                real_t backward(const index_t thread_index,
                                const index_t input_index,
                                const NNet::State &state,
                                const LossType auto &loss) {
                    const auto target = _targets[input_index];
                    const auto neurons = _view.neurons();
                    const auto connections = _view.connections();
                    const auto partitions = _view.partitions();
                    const auto layers = _view.layers();
                    const auto partition_layer_indices = _view.partition_layer_indices();
                    const auto output_neurons_indices = _view.output_neurons_indices();
                    const auto signals = state.signals();
                    const auto weighted_sums = state.weighted_sums();
                    auto &deltas = _deltas[thread_index];
                    auto &gradient = _gradients[thread_index];
                    std::ranges::fill(deltas, 0.0_real);
                    // compute loss and output deltas
                    real_t loss_value{0.0_real};
                    for (const auto &[target_value, output_neuron_index] : std::views::zip(target, output_neurons_indices)) {
                        const auto &signal = signals[output_neuron_index];
                        const auto &weighted_sum = weighted_sums[output_neuron_index];
                        deltas[output_neuron_index] = loss.derivative(target_value, signal) * neurons[output_neuron_index].activation()->derivative(weighted_sum);
                        loss_value += loss(target_value, signal);
                    }
                    // walk backwards and compute other deltas (partitions are in topological order)
                    for (auto partition_index = std::ssize(partitions) - 1; partition_index >= 0; --partition_index) {
                        const auto layer_index = partition_layer_indices[partition_index];
                        if (layer_index != net::DAGNet::Layer::no_layer) {
                            const auto &layer = layers[layer_index];
                            for (auto row_index{0}; row_index < layer.rx_count; ++row_index) {
                                const auto rx_index = layer.rx_begin + row_index;
                                if (neurons[rx_index].type() != NeuronType::output) {
                                    const auto &weighted_sum = weighted_sums[rx_index];
                                    deltas[rx_index] *= neurons[rx_index].activation()->derivative(weighted_sum);
                                }
                                const auto row_offset = layer.weight_offset + (row_index * layer.tx_count);
                                for (auto column_index{0}; column_index < layer.tx_count; ++column_index) {
                                    const auto tx_index = layer.tx_begin + column_index;
                                    const auto connection_index = row_offset + column_index;
                                    deltas[tx_index] += deltas[rx_index] * _weights[connection_index];
                                }
                            }
                            partition_index -= layer.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[partition_index];
                            const auto &rx_index = partition.rx_index;
                            if (neurons[rx_index].type() != NeuronType::output) {
                                const auto &weighted_sum = weighted_sums[rx_index];
                                deltas[rx_index] *= neurons[rx_index].activation()->derivative(weighted_sum);
                            }
                            for (const auto &connection_index : std::views::iota(partition.connection_begin, partition.connection_end)) {
                                const auto tx_index = connections[connection_index].tx_index;
                                deltas[tx_index] += deltas[rx_index] * _weights[connection_index];
                            }
                        }
                    }
                    // compute gradient (sum gradient since a thread could process multiple samples)
                    for (index_t partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                        const auto layer_index = partition_layer_indices[partition_index];
                        if (layer_index != net::DAGNet::Layer::no_layer) {
                            const auto &layer = layers[layer_index];
                            for (auto row_index{0}; row_index < layer.rx_count; ++row_index) {
                                const auto rx_index = layer.rx_begin + row_index;
                                const auto row_offset = layer.weight_offset + (row_index * layer.tx_count);
                                for (auto column_index{0}; column_index < layer.tx_count; ++column_index) {
                                    const auto tx_index = layer.tx_begin + column_index;
                                    const auto connection_index = row_offset + column_index;
                                    gradient[connection_index] += deltas[rx_index] * signals[tx_index];
                                }
                            }
                            partition_index += layer.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[partition_index];
                            for (const auto &connection_index : std::views::iota(partition.connection_begin, partition.connection_end)) {
                                const auto &[tx_index, rx_index] = connections[connection_index];
                                gradient[connection_index] += deltas[rx_index] * signals[tx_index];
                            }
                        }
                    }
                    return loss_value;
                }
            private:
                // Data members
                net::DAGNet::View _view{};
                std::vector<std::span<const real_t>> _targets{};
                std::vector<real_t> _weights{};
                dataset_vector_t _deltas{};
                dataset_vector_t _gradients{};
                std::vector<real_t> _batch_gradient{};
                Optimizer _optimizer;
                dag::update::omp::Backend _update_backend{};
        };
}
