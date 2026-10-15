#pragma once
#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>
#include <omp.h>
#include "hnnet/loss_type.h"
#include "hnnet/builtin/nets/dag_net.h"
#include "hnnet/builtin/backends/omp/backprop/optimizer_type.h"

namespace hnnet::builtin::backend::omp::backprop {
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
                // Perform forward and backward passes for a batch of data
                real_t forward_backward(net::DAGNet &net,
                                        const index_t batch_begin,
                                        const index_t batch_end,
                                        const LossType auto &loss_function) {
                    auto view = net.view();
                    const auto batch_size = batch_end - batch_begin;
                    const auto parallel = batch_size > 1;
                    const auto thread_count = parallel ? std::ssize(_states) : index_t{1};
                    for (auto tid{0}; tid < thread_count; ++tid) {
                        std::ranges::fill(_gradients[tid], 0.0_real);
                    }
                    real_t loss{0.0_real};
                    #pragma omp parallel for if(parallel) reduction(+:loss)
                    for (auto isample = batch_begin; isample < batch_end; ++isample) {
                        const auto tid = omp_get_thread_num();
                        net.update(_states[tid], _inputs[isample]);
                        loss += backward(view, tid, isample, loss_function);
                    }
                    if (parallel) {
                        std::ranges::fill(_batch_gradient, 0.0_real);
                        for (auto tid{0}; tid < thread_count; ++tid) {
                            const auto &gradient = _gradients[tid];
                            for (auto iconn{0}; iconn < view.connection_count(); ++iconn) {
                                _batch_gradient[iconn] += gradient[iconn] / batch_size;
                            }
                        }
                    }
                    else {
                        std::ranges::copy(_gradients.front(), _batch_gradient.begin());
                    }
                    return loss;
                }
                // Transfer the net to host memory (if needed)
                void to_host(net::DAGNet &) {}
            protected:
                // Compute the error and delta weights contribution of a single sample
                real_t backward(net::DAGNet::View &view,
                                const index_t tid,
                                const index_t isample,
                                const LossType auto &loss_function) {
                    auto &state = _states[tid];
                    auto &deltas = _deltas[tid];
                    auto &gradient = _gradients[tid];
                    const auto targets = _targets[isample];
                    const auto neurons = view.neurons();
                    const auto connections = view.connections();
                    const auto weights = view.weights();
                    const auto partitions = view.partitions();
                    const auto output_neurons = view.iout_neurons();
                    std::ranges::fill(deltas, 0.0_real);
                    // compute loss and output deltas
                    real_t loss{0.0_real};
                    for (const auto &[target, iout] : std::views::zip(targets, output_neurons)) {
                        const auto signal = state.signals[iout];
                        loss += loss_function(target, signal);
                        deltas[iout] = loss_function.derivative(target, signal) * neurons[iout].activation()->derivative(state.weighted_sums[iout]);
                    }
                    // walk backwards and compute other deltas (partitions are in topological order)
                    for (auto ipart = std::ssize(partitions) - 1; ipart >= 0; --ipart) {
                        if (view.is_dense(ipart)) {
                            const auto &block = view.dense_block(ipart);
                            for (auto irow{0}; irow < block.rx_count; ++irow) {
                                const auto irx = block.rx_begin + irow;
                                const auto &neuron = neurons[irx];
                                auto &delta = deltas[irx];
                                if (neuron.type() != NeuronType::output) {
                                    delta *= neuron.activation()->derivative(state.weighted_sums[irx]);
                                }
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                for (auto icol{0}; icol < block.tx_count; ++icol) {
                                    deltas[block.tx_begin + icol] += delta * weights[row_offset + icol];
                                }
                            }
                            ipart -= block.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[ipart];
                            const auto &neuron = neurons[partition.irx];
                            auto &delta = deltas[partition.irx];
                            if (neuron.type() != NeuronType::output) {
                                delta *= neuron.activation()->derivative(state.weighted_sums[partition.irx]);
                            }
                            for (const auto &iconn : std::views::iota(partition.conn_begin, partition.conn_end)) {
                                const auto itx = connections[iconn].itx;
                                deltas[itx] += delta * weights[iconn];
                            }
                        }
                    }
                    // compute gradient (sum gradient since a thread could process multiple samples)
                    for (auto ipart{0}; ipart < std::ssize(partitions); ++ipart) {
                        if (view.is_dense(ipart)) {
                            const auto &block = view.dense_block(ipart);
                            for (auto irow{0}; irow < block.rx_count; ++irow) {
                                const auto delta = deltas[block.rx_begin + irow];
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                for (auto icol{0}; icol < block.tx_count; ++icol) {
                                    gradient[row_offset + icol] += delta * state.signals[block.tx_begin + icol];
                                }
                            }
                            ipart += block.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[ipart];
                            for (const auto &iconn : std::views::iota(partition.conn_begin, partition.conn_end)) {
                                const auto irx = connections[iconn].irx;
                                const auto itx = connections[iconn].itx;
                                gradient[iconn] += deltas[irx] * state.signals[itx];
                            }
                        }
                    }
                    return loss;
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
