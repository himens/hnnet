#pragma once
#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>
#include <omp.h>
#include "optimizer_type.h"
#include "hnnet/loss_type.h"
#include "hnnet/nets/dag_net.h"

namespace hnnet::backprop::omp {
    using delta_vector_t = std::vector<real_t>;
    using grad_vector_t = std::vector<real_t>;
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
                void init(net::DAGNet &net) {
                    auto view = net.view();
                    const auto neuron_count = view.neuron_count();
                    const auto connection_count = view.connection_count();
                    const auto max_threads = omp_get_max_threads();
                    _states.assign(max_threads, NNet::State{neuron_count});
                    _deltas.assign(max_threads, delta_vector_t(neuron_count, 0.0_real));
                    _gradients.assign(max_threads, grad_vector_t(connection_count, 0.0_real));
                    _batch_gradient.assign(connection_count, 0.0_real);
                    _optimizer.init(connection_count);
                }
                // Transfer the network and data to the device
                void to_device(net::DAGNet &net, const DatasetType auto &inputs, const DatasetType auto &targets) {
                    _weights = net.view().weights();
                    _inputs.clear();
                    _targets.clear();
                    for (const auto &sample : inputs) {
                        _inputs.emplace_back(sample);
                    }
                    for (const auto &sample : targets) {
                        _targets.emplace_back(sample);
                    }
                }
                // Perform backprop (forward-backward plus weights update)
                real_t backprop(net::DAGNet &net,
                                const index_t batch_begin,
                                const index_t batch_end,
                                const LossType auto &loss) {
                    auto view = net.view();
                    const auto batch_size = batch_end - batch_begin;
                    const auto parallel = batch_size > 1;
                    const auto thread_count = parallel ? std::ssize(_states) : index_t{1};
                    for (auto thread_index{0}; thread_index < thread_count; ++thread_index) {
                        std::ranges::fill(_gradients[thread_index], 0.0_real);
                    }
                    real_t loss_value{0.0_real};
                    #pragma omp parallel for if(parallel) reduction(+:loss_value)
                    for (auto sample_index = batch_begin; sample_index < batch_end; ++sample_index) {
                        const auto thread_index = omp_get_thread_num();
                        net.update(_states[thread_index], _inputs[sample_index]);
                        loss_value += backward(view, thread_index, sample_index, loss);
                    }
                    if (parallel) {
                        std::ranges::fill(_batch_gradient, 0.0_real);
                        for (auto thread_index{0}; thread_index < thread_count; ++thread_index) {
                            const auto &gradient = _gradients[thread_index];
                            for (auto connection_index{0}; connection_index < view.connection_count(); ++connection_index) {
                                _batch_gradient[connection_index] += gradient[connection_index] / batch_size;
                            }
                        }
                    }
                    _optimizer.update(_weights, parallel ? _batch_gradient : _gradients.front());
                    return loss_value;
                }
                // Transfer the net to host memory (if needed)
                void to_host(net::DAGNet &) {}
            protected:
                // Compute the error and delta weights contribution of a single sample
                real_t backward(net::DAGNet::View &view,
                                const index_t thread_index,
                                const index_t sample_index,
                                const LossType auto &loss) {
                    const auto targets = _targets[sample_index];
                    const auto neurons = view.neurons();
                    const auto connections = view.connections();
                    const auto weights = view.weights();
                    const auto partitions = view.partitions();
                    const auto output_neurons_indices = view.output_neurons_indices();
                    auto &state = _states[thread_index];
                    auto &deltas = _deltas[thread_index];
                    auto &gradient = _gradients[thread_index];
                    std::ranges::fill(deltas, 0.0_real);
                    // compute loss and output deltas
                    real_t loss_value{0.0_real};
                    for (const auto &[target, neuron_index] : std::views::zip(targets, output_neurons_indices)) {
                        const auto &signal = state.signals[neuron_index];
                        const auto &weighted_sum = state.weighted_sums[neuron_index];
                        deltas[neuron_index] = loss.derivative(target, signal) * neurons[neuron_index].activation()->derivative(weighted_sum);
                        loss_value += loss(target, signal);
                    }
                    // walk backwards and compute other deltas (partitions are in topological order)
                    for (auto partition_index = std::ssize(partitions) - 1; partition_index >= 0; --partition_index) {
                        if (view.is_dense(partition_index)) {
                            const auto &block = view.dense_block(partition_index);
                            for (auto row_index{0}; row_index < block.rx_count; ++row_index) {
                                const auto rx_index = block.rx_begin + row_index;
                                if (neurons[rx_index].type() != NeuronType::output) {
                                    const auto &weighted_sum = state.weighted_sums[rx_index];
                                    deltas[rx_index] *= neurons[rx_index].activation()->derivative(weighted_sum);
                                }
                                const auto row_offset = block.weight_offset + (row_index * block.tx_count);
                                for (auto column_index{0}; column_index < block.tx_count; ++column_index) {
                                    const auto tx_index = block.tx_begin + column_index;
                                    const auto connection_index = row_offset + column_index;
                                    deltas[tx_index] += deltas[rx_index] * weights[connection_index];
                                }
                            }
                            partition_index -= block.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[partition_index];
                            const auto &rx_index = partition.rx_index;
                            if (neurons[rx_index].type() != NeuronType::output) {
                                const auto &weighted_sum = state.weighted_sums[rx_index];
                                deltas[rx_index] *= neurons[rx_index].activation()->derivative(weighted_sum);
                            }
                            for (const auto &connection_index : std::views::iota(partition.connection_begin, partition.connection_end)) {
                                const auto tx_index = connections[connection_index].tx_index;
                                deltas[tx_index] += deltas[rx_index] * weights[connection_index];
                            }
                        }
                    }
                    // compute gradient (sum gradient since a thread could process multiple samples)
                    for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                        if (view.is_dense(partition_index)) {
                            const auto &block = view.dense_block(partition_index);
                            for (auto row_index{0}; row_index < block.rx_count; ++row_index) {
                                const auto rx_index = block.rx_begin + row_index;
                                const auto row_offset = block.weight_offset + (row_index * block.tx_count);
                                for (auto column_index{0}; column_index < block.tx_count; ++column_index) {
                                    const auto tx_index = block.tx_begin + column_index;
                                    const auto connection_index = row_offset + column_index;
                                    gradient[connection_index] += deltas[rx_index] * state.signals[tx_index];
                                }
                            }
                            partition_index += block.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[partition_index];
                            for (const auto &connection_index : std::views::iota(partition.connection_begin, partition.connection_end)) {
                                const auto rx_index = connections[connection_index].rx_index;
                                const auto tx_index = connections[connection_index].tx_index;
                                gradient[connection_index] += deltas[rx_index] * state.signals[tx_index];
                            }
                        }
                    }
                    return loss_value;
                }
            protected:
                // Data members
                std::vector<std::span<const real_t>> _inputs{};
                std::vector<std::span<const real_t>> _targets{};
                std::span<real_t> _weights{};
                std::vector<NNet::State> _states{};
                std::vector<delta_vector_t> _deltas{};
                std::vector<grad_vector_t> _gradients{};
                grad_vector_t _batch_gradient{};
                Optimizer _optimizer;
        };
}
