#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    // Mean squared error loss: only knows a single (target, signal) pair, nothing about the net topology.
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
    // Plain SGD with momentum: owns the per-batch update logic and its persistent state (previous update, for momentum).
    class SGDMomentum {
        public:
            explicit SGDMomentum(const real_t learning_rate = 0.0, const real_t momentum = 0.0) : _learning_rate(learning_rate), _momentum(momentum) {}
            template <typename View>
                void apply(View &view, const std::vector<real_t> &batch_deltas, const real_t batch_size) {
                    if (_prev_update.empty()) {
                        _prev_update.assign(batch_deltas.size(), 0.0);
                    }
                    for (index_t iconn{0}; iconn < std::ssize(batch_deltas); ++iconn) {
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
    // Back-propagation learning rule: owns the whole epoch, splitting it into mini-batches
    // and processing the samples of each mini-batch in parallel (one NNetState per thread).
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
                        std::vector<NNetState> states;
                        states.reserve(num_threads);
                        for (index_t t{0}; t < num_threads; ++t) {
                            states.emplace_back(neuron_count);
                        }
                        std::vector<std::vector<real_t>> backward_deltas(num_threads, std::vector<real_t>(neuron_count, 0.0));
                        std::vector<std::vector<real_t>> thread_deltas(num_threads, std::vector<real_t>(connection_count, 0.0));

                        real_t total_error{0.0};
                        for (index_t ibatch{0}; ibatch < std::ssize(samples); ibatch += _batch_size) {
                            const auto batch_end = std::min<index_t>(samples.size(), ibatch + _batch_size);
                            for (auto &deltas : thread_deltas) {
                                std::ranges::fill(deltas, 0.0);
                            }
                            #pragma omp parallel for reduction(+:total_error)
                            for (auto i = ibatch; i < batch_end; ++i) {
                                const auto tid = omp_get_thread_num();
                                auto &state = states[tid];
                                state.reset();
                                net.inject(state, samples[i].inputs);
                                net.broadcast(state);
                                total_error += backward(net, state, samples[i].targets, backward_deltas[tid], thread_deltas[tid]);
                            }
                            // sequential reduction: sum every thread's contribution into a single per-connection buffer
                            std::vector<real_t> batch_deltas(connection_count, 0.0);
                            for (const auto &deltas : thread_deltas) {
                                for (index_t iconn{0}; iconn < connection_count; ++iconn) {
                                    batch_deltas[iconn] += deltas[iconn];
                                }
                            }
                            // single, sequential weight update
                            auto view = net.view();
                            _optimizer.apply(view, batch_deltas, static_cast<real_t>(batch_end - ibatch));
                        }
                        return total_error / samples.size();
                    }
            private:
                // Compute the error and per-connection delta_weight contribution of a single sample (weights untouched)
                template <NNetType Net>
                    real_t backward(Net &net, NNetState &state, const output_t<Net> &targets,
                                     std::vector<real_t> &deltas, std::vector<real_t> &out_delta_weights) const {
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
                        for (index_t ipart{0}; ipart < std::ssize(reversed_partitions); ipart++) {
                            const auto &partition = reversed_partitions[ipart];
                            const auto iblock = partition.iblock;
                            if (iblock != Net::no_block) {
                                const auto &block = view.dense_block(iblock);
                                for (index_t irow{0}; irow < block.rx_count; ++irow) {
                                    const auto irx = block.irx_begin + irow;
                                    const auto &rx = view.neuron(irx);
                                    auto &delta_rx = deltas[irx];
                                    if (rx.type() != NeuronType::output) {
                                        delta_rx *= rx.activation()->derivative(state.weighted_sums[irx]);
                                    }
                                    const auto row_offset = block.weight_offset + irow * block.tx_count;
                                    for (index_t icol{0}; icol < block.tx_count; ++icol) {
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
                        for (index_t ipart{0}; ipart < std::ssize(partitions); ipart++) {
                            const auto &partition = partitions[ipart];
                            const auto iblock = partition.iblock;
                            if (iblock != Net::no_block) {
                                const auto &block = view.dense_block(iblock);
                                for (index_t irow{0}; irow < block.rx_count; ++irow) {
                                    const auto delta_rx = deltas[block.irx_begin + irow];
                                    const auto row_offset = block.weight_offset + irow * block.tx_count;
                                    for (index_t icol{0}; icol < block.tx_count; ++icol) {
                                        out_delta_weights[row_offset + icol] = delta_rx * state.signals[block.itx_begin + icol];
                                    }
                                }
                                ipart += block.rx_count - 1;
                                continue;
                            }
                            for (const auto &iconn : std::views::iota(partition.iconn_begin, partition.iconn_end)) {
                                const auto irx = view.connection(iconn).irx;
                                const auto itx = view.connection(iconn).itx;
                                out_delta_weights[iconn] = deltas[irx] * state.signals[itx];
                            }
                        }
                        return total_error;
                    }
                // Data members
                index_t _batch_size{1};
                Loss _loss;
                Optimizer _optimizer;
        };
}

