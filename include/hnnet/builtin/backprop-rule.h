#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    ///////////////////
    // MSELoss class //
    ///////////////////
    class MSELoss {
        public:
            real_t value(const real_t target, const real_t signal) const {
                const auto error = target - signal;
                return error * error;
            }
            // Seed value fed to the output neuron's activation derivative (matches the classic error term).
            real_t derivative(const real_t target, const real_t signal) const {
                return target - signal;
            }
    };
    ///////////////////////
    // SGDMomentum class //
    ///////////////////////
    class SGDMomentum {
        public:
            explicit SGDMomentum(const real_t learning_rate = 0.0, const real_t momentum = 0.0) : _learning_rate(learning_rate), _momentum(momentum) {}
            template <typename View>
                void apply(View &view, const std::vector<real_t> &batch_deltas, const real_t batch_size) {
                    if (_prev_update.empty()) {
                        _prev_update.assign(batch_deltas.size(), 0.0);
                    }
                    for (auto iconn{0}; iconn < std::ssize(batch_deltas); ++iconn) {
                        const auto dweight = (_learning_rate * batch_deltas[iconn] / batch_size) + (_momentum * _prev_update[iconn]);
                        view.weight(iconn) += dweight;
                        _prev_update[iconn] = dweight;
                    }
                }
        private:
            real_t _learning_rate;
            real_t _momentum;
            std::vector<real_t> _prev_update{};
    };
    ////////////////////////
    // BackpropRule class //
    ////////////////////////
    // Back-propagation learning rule: split samples into mini-batches and process batch samples in parallel
    template <typename Loss = MSELoss, typename Optimizer = SGDMomentum>
        class BackpropRule {
            public:
                // Constructor
                explicit BackpropRule(const real_t learning_rate, const real_t momentum = 0.0, const int_t batch_size = 1, Loss loss = {})
                    : _batch_size(batch_size), _loss(std::move(loss)), _optimizer(learning_rate, momentum) {}
                // Learn from a whole epoch of training samples
                template <NNetType Net>
                    real_t learn(Net &net, const std::vector<typename Net::TrainingData> &samples) {
                        const auto neuron_count = net.view().neuron_count();
                        const auto connection_count = net.view().connection_count();
                        const auto num_threads = omp_get_max_threads();
                        if (_states.empty()) {  // lazy init: allocate once, reuse across every epoch
                            _states.reserve(num_threads);
                            for (auto t{0}; t < num_threads; ++t) {
                                _states.emplace_back(neuron_count);
                            }
                            _backward_deltas.assign(num_threads, std::vector<real_t>(neuron_count, 0.0));
                            _thread_deltas.assign(num_threads, std::vector<real_t>(connection_count, 0.0));
                            _batch_deltas.assign(connection_count, 0.0);
                        }
                        real_t total_error{0.0};
                        for (auto ibatch{0}; ibatch < std::ssize(samples); ibatch += _batch_size) {
                            const auto batch_end = std::min<index_t>(samples.size(), ibatch + _batch_size);
                            const auto batch_count = batch_end - ibatch;
                            // below this size, opening a parallel region isn't worth its fork/join overhead: the if() clause runs the loop sequentially instead
                            const auto parallel = batch_count >= num_threads;
                            const auto active_threads = parallel ? num_threads : index_t{1};
                            for (auto t{0}; t < active_threads; ++t) {
                                std::ranges::fill(_thread_deltas[t], 0.0);
                            }
                            #pragma omp parallel for if(parallel) reduction(+:total_error)
                            for (auto i = ibatch; i < batch_end; ++i) {
                                const auto tid = omp_get_thread_num();
                                auto &state = _states[tid];
                                state.reset();
                                net.inject(state, samples[i].inputs);
                                net.broadcast(state);
                                total_error += backward(net, state, samples[i].targets, _backward_deltas[tid], _thread_deltas[tid]);
                            }
                            // sequential reduction: sum every actually-used thread's contribution into a single per-connection buffer
                            std::ranges::fill(_batch_deltas, 0.0);
                            for (auto t{0}; t < active_threads; ++t) {
                                const auto &deltas = _thread_deltas[t];
                                for (auto iconn{0}; iconn < connection_count; ++iconn) {
                                    _batch_deltas[iconn] += deltas[iconn];
                                }
                            }
                            // single, sequential weight update
                            auto view = net.view();
                            _optimizer.apply(view, _batch_deltas, static_cast<real_t>(batch_count));
                        }
                        return total_error / samples.size();
                    }
            private:
                // Compute the error and delta weights contribution of a single sample
                template <NNetType Net>
                    real_t backward(Net &net, NNetState &state, const output_t<Net> &targets, std::vector<real_t> &deltas, std::vector<real_t> &dweights) const {
                        auto view = net.view();
                        std::ranges::fill(deltas, 0.0);
                        // seed output deltas using the loss
                        real_t total_error{0.0};
                        for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                            const auto signal = state.signals[iout];
                            total_error += _loss.value(target, signal);
                            deltas[iout] = _loss.derivative(target, signal) * view.neuron(iout).activation()->derivative(state.weighted_sums[iout]);
                        }
                        // partitions are already in topological order: walk them backwards
                        const auto partitions = view.partitions();
                        auto reversed_partitions = partitions | std::views::reverse;
                        for (auto ipart{0}; ipart < std::ssize(reversed_partitions); ipart++) {
                            const auto &partition = reversed_partitions[ipart];
                            const auto iblock = partition.iblock;
                            if (iblock != Net::no_block) {
                                const auto &block = view.dense_block(iblock);
                                for (auto irow{0}; irow < block.rx_count; ++irow) {
                                    const auto irx = block.irx_begin + irow;
                                    const auto &rx = view.neuron(irx);
                                    auto &delta_rx = deltas[irx];
                                    if (rx.type() != NeuronType::output) {
                                        delta_rx *= rx.activation()->derivative(state.weighted_sums[irx]);
                                    }
                                    const auto row_offset = block.weight_offset + irow * block.tx_count;
                                    for (auto icol{0}; icol < block.tx_count; ++icol) {
                                        deltas[block.itx_begin + icol] += delta_rx * view.weight(row_offset + icol);
                                    }
                                }
                                ipart += block.rx_count - 1;
                                continue;
                            }
                            const auto &rx = view.neuron(partition.irx);
                            auto &delta_rx = deltas[partition.irx];
                            if (rx.type() != NeuronType::output) {
                                delta_rx *= rx.activation()->derivative(state.weighted_sums[partition.irx]);
                            }
                            for (const auto &iconn : std::views::iota(partition.iconn_begin, partition.iconn_end)) {
                                const auto itx = view.connection(iconn).itx;
                                deltas[itx] += delta_rx * view.weight(iconn);
                            }
                        }
                        // per-connection delta_weight (pure, no learning rate/momentum): each connection written exactly once
                        for (auto ipart{0}; ipart < std::ssize(partitions); ipart++) {
                            const auto &partition = partitions[ipart];
                            const auto iblock = partition.iblock;
                            if (iblock != Net::no_block) {
                                const auto &block = view.dense_block(iblock);
                                for (auto irow{0}; irow < block.rx_count; ++irow) {
                                    const auto delta_rx = deltas[block.irx_begin + irow];
                                    const auto row_offset = block.weight_offset + irow * block.tx_count;
                                    for (auto icol{0}; icol < block.tx_count; ++icol) {
                                        dweights[row_offset + icol] = delta_rx * state.signals[block.itx_begin + icol];
                                    }
                                }
                                ipart += block.rx_count - 1;
                                continue;
                            }
                            for (const auto &iconn : std::views::iota(partition.iconn_begin, partition.iconn_end)) {
                                const auto irx = view.connection(iconn).irx;
                                const auto itx = view.connection(iconn).itx;
                                dweights[iconn] = deltas[irx] * state.signals[itx];
                            }
                        }
                        return total_error;
                    }
                // Data members
                int_t _batch_size{1};
                Loss _loss;
                Optimizer _optimizer;
                std::vector<NNetState> _states{};
                std::vector<std::vector<real_t>> _backward_deltas{};
                std::vector<std::vector<real_t>> _thread_deltas{};
                std::vector<real_t> _batch_deltas{};
        };
}

