#pragma once
#include "hnnet/builtin/backprop/backend.h"
#include "hnnet/builtin/backprop/optimizer.h"
#include "hnnet/builtin/dag_net.h"

namespace hnnet::builtin {
    using delta_vector_t = std::vector<real_t>;
    //////////////////
    // OpenMP class //
    //////////////////
    class OpenMPBackend {
            public:
                // Transfer the network and data to the device
                void to_device(DAGNet &net, const DatasetType auto &, const DatasetType auto &) {
                    auto view = net.view();
                    const auto neuron_count = view.neuron_count();
                    const auto connection_count = view.connection_count();
                    const auto max_threads = omp_get_max_threads();
                    if (std::ssize(_states) != max_threads or
                        std::ssize(_states.front().signals) != neuron_count or
                        std::ssize(_batch_gradient) != connection_count) {
                        _states.assign(max_threads, NNetState{neuron_count});
                        _deltas.assign(max_threads, delta_vector_t(neuron_count, 0.0_real));
                        _gradients.assign(max_threads, grad_vector_t(connection_count, 0.0_real));
                        _batch_gradient.assign(connection_count, 0.0_real);
                    }
                }
                // Perform forward and backward passes for a batch of data
                real_t forward_backward(DAGNet &net, const DatasetType auto &inputs,
                                        const DatasetType auto &targets, const index_t batch_begin,
                                        const index_t batch_end, const LossType auto &loss_function) {
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
                        auto &state = _states[tid];
                        net.update(state, inputs[isample]);
                        loss += backward(view, state, targets[isample], _deltas[tid], _gradients[tid], loss_function);
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
                void update_weights(DAGNet &net, OptimizerType auto &optimizer) {
                    auto view = net.view();
                    optimizer.update(view, _batch_gradient);
                }
                void to_host(DAGNet &) {}
            private:
                // Compute the error and delta weights contribution of a single sample
                real_t backward(DAGNet::View &view,
                                NNetState &state,
                                const DataType auto &targets,
                                delta_vector_t &deltas,
                                grad_vector_t &gradient,
                                const LossType auto &loss_function) const {
                    std::ranges::fill(deltas, 0.0_real);
                    // compute loss and output deltas
                    real_t loss{0.0_real};
                    for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                        const auto signal = state.signals[iout];
                        loss += loss_function(target, signal);
                        deltas[iout] = loss_function.derivative(target, signal) * view.neuron(iout).activation()->derivative(state.weighted_sums[iout]);
                    }
                    // walk backwards and compute other deltas (partitions are in topological order)
                    const auto partitions = view.partitions();
                    for (auto ipart = std::ssize(partitions) - 1; ipart >= 0; --ipart) {
                        if (view.is_dense(ipart)) {
                            const auto &block = view.dense_block(ipart);
                            for (auto irow{0}; irow < block.rx_count; ++irow) {
                                const auto irx = block.rx_begin + irow;
                                const auto &rx = view.neuron(irx);
                                auto &delta = deltas[irx];
                                if (rx.type() != NeuronType::output) { // si puo' eliminare?
                                    delta *= rx.activation()->derivative(state.weighted_sums[irx]);
                                }
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                for (auto icol{0}; icol < block.tx_count; ++icol) {
                                    deltas[block.tx_begin + icol] += delta * view.weight(row_offset + icol);
                                }
                            }
                            ipart -= block.rx_count - 1;
                        }
                        else {
                            const auto &partition = partitions[ipart];
                            const auto &rx = view.neuron(partition.irx);
                            auto &delta = deltas[partition.irx];
                            if (rx.type() != NeuronType::output) { // si puo' eliminare?
                                delta *= rx.activation()->derivative(state.weighted_sums[partition.irx]);
                            }
                            for (const auto &iconn : std::views::iota(partition.conn_begin, partition.conn_end)) {
                                const auto itx = view.connection(iconn).itx;
                                deltas[itx] += delta * view.weight(iconn);
                            }
                        }
                    }
                    // compute gradient (sum gradient since a thread could process multiple samples)
                    for (auto ipart{0}; ipart < std::ssize(partitions); ipart++) {
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
                                const auto irx = view.connection(iconn).irx;
                                const auto itx = view.connection(iconn).itx;
                                gradient[iconn] += deltas[irx] * state.signals[itx];
                            }
                        }
                    }
                    return loss;
                }
            private:
                std::vector<NNetState> _states{};
                std::vector<delta_vector_t> _deltas{};
                std::vector<grad_vector_t> _gradients{};
                grad_vector_t _batch_gradient{};
    };
}