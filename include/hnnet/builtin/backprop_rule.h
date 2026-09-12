#pragma once
#include "hnnet/nnet.h"
#include "hnnet/builtin/losses.h"

namespace hNNet::Builtin {
    template <typename T, typename View>
        concept OptimizerType = requires (T &optimizer, View &view, const std::vector<real_t> &batch_dweights, const real_t batch_size, const real_t learning_rate) {
            optimizer.apply(view, batch_dweights, batch_size, learning_rate);
        };
    ///////////////////////
    // SGDMomentum class //
    ///////////////////////
    class SGDMomentum {
        public:
            explicit SGDMomentum(const real_t momentum = 0.0)
                : _momentum(momentum) {
                    if (_momentum < 0.0) {
                        throw std::invalid_argument("SGDMomentum::SGDMomentum: momentum must be >= 0");
                    }
                    std::println("SGDMomentum::SGDMomentum: momentum: {}", _momentum);
                }
            template <typename View>
                void apply(View &view, const std::vector<real_t> &batch_dweights, const real_t batch_size, const real_t learning_rate) {
                    if (_prev_dweights.empty()) {
                        _prev_dweights.assign(batch_dweights.size(), 0.0);
                    }
                    for (auto iconn{0}; iconn < std::ssize(batch_dweights); ++iconn) {
                        const auto dweight = (learning_rate * batch_dweights[iconn] / batch_size) + (_momentum * _prev_dweights[iconn]);
                        view.weight(iconn) += dweight;
                        _prev_dweights[iconn] = dweight;
                    }
                }
        private:
            real_t _momentum;
            std::vector<real_t> _prev_dweights{};
    };
    ////////////////////////
    // BackpropRule class //
    ////////////////////////
    // Back-propagation learning rule: split samples into mini-batches and process batch samples in parallel
    template <typename Optimizer = SGDMomentum, LossType Loss = MSELoss>
        class BackpropRule {
            public:
                // Constructor
                explicit BackpropRule(const real_t learning_rate, Optimizer optimizer = Optimizer{}, const int_t batch_size = 1, Loss loss = {})
                    : _learning_rate(learning_rate),_optimizer(std::move(optimizer)), _batch_size(batch_size), _loss(std::move(loss)) {
                        if (_learning_rate < 0.0) {
                            throw std::invalid_argument("BackpropRule::BackpropRule: learning_rate must be >= 0");
                        }
                        if (batch_size <= 0) {
                            throw std::invalid_argument("BackpropRule::BackpropRule: batch_size must be > 0");
                        }
                        std::println("BackpropRule:learn: learning rate: {}, batch size: {}", _learning_rate, _batch_size);
                    }
                // Learn from a whole epoch of training samples
                template <NNetType Net>
                    requires OptimizerType<Optimizer, typename Net::View>
                    real_t learn(Net &net, const std::span<typename Net::TrainingData> samples) {
                        const auto neuron_count = net.view().neuron_count();
                        const auto connection_count = net.view().connection_count();
                        const auto max_threads = omp_get_max_threads();
                        if (_states.empty()) {  // lazy init: allocate once, reuse across every epoch
                            _states.reserve(max_threads);
                            for (auto tid{0}; tid < max_threads; ++tid) {
                                _states.emplace_back(neuron_count);
                            }
                            _deltas.assign(max_threads, std::vector<real_t>(neuron_count, 0.0));
                            _thread_dweights.assign(max_threads, std::vector<real_t>(connection_count, 0.0));
                            _batch_dweights.assign(connection_count, 0.0);
                        }
                        //std::ranges::shuffle(samples, random_generator());
                        real_t loss{0.0};
                        const auto batch_count = static_cast<int_t>(std::ceil(static_cast<real_t>(samples.size()) / _batch_size));
                        for (auto ibatch{0}; ibatch < batch_count; ++ibatch) {
                            const auto batch_begin = ibatch * _batch_size;
                            const auto batch_end = std::min<index_t>(samples.size(), batch_begin + _batch_size);
                            const auto batch_size = batch_end - batch_begin;
                            const auto parallel = batch_size >= max_threads;
                            const auto thread_count = parallel ? max_threads : index_t{1};
                            for (auto tid{0}; tid < thread_count; ++tid) {
                                std::ranges::fill(_thread_dweights[tid], 0.0);
                            }
                            #pragma omp parallel for if(parallel) reduction(+:loss)
                            for (auto isample = batch_begin; isample < batch_end; ++isample) {
                                const auto tid = omp_get_thread_num();
                                auto &state = _states[tid];
                                const auto &sample = samples[isample];
                                net.inject(state, sample.inputs);
                                net.broadcast(state);
                                loss += backward(net, state, sample.targets, _deltas[tid], _thread_dweights[tid]);
                            }
                            // single, sequential weight update
                            auto view = net.view();
                            if (thread_count == 1) {
                                _optimizer.apply(view, _thread_dweights[0], static_cast<real_t>(batch_size), _learning_rate);
                            }
                            else {
                                // sequential reduction: sum contribution of each thread
                                std::ranges::fill(_batch_dweights, 0.0);
                                for (auto tid{0}; tid < thread_count; ++tid) {
                                    const auto &dweights = _thread_dweights[tid];
                                    for (auto iconn{0}; iconn < connection_count; ++iconn) {
                                        _batch_dweights[iconn] += dweights[iconn];
                                    }
                                }
                                _optimizer.apply(view, _batch_dweights, static_cast<real_t>(batch_size), _learning_rate);
                            }
                        }
                        return loss / samples.size();
                    }
            private:
                // Compute the error and delta weights contribution of a single sample
                template <NNetType Net>
                    real_t backward(Net &net, NNetState &state, const output_t<Net> &targets, std::vector<real_t> &deltas, std::vector<real_t> &dweights) const {
                        auto view = net.view();
                        std::ranges::fill(deltas, 0.0);
                        // seed output deltas using the loss
                        real_t loss{0.0};
                        for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                            const auto signal = state.signals[iout];
                            loss += _loss(target, signal);
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
                        // per-connection delta_weight (pure, no learning rate/momentum)
                        for (auto ipart{0}; ipart < std::ssize(partitions); ipart++) {
                            const auto &partition = partitions[ipart];
                            const auto iblock = partition.iblock;
                            if (iblock != Net::no_block) {
                                const auto &block = view.dense_block(iblock);
                                for (auto irow{0}; irow < block.rx_count; ++irow) {
                                    const auto delta_rx = deltas[block.irx_begin + irow];
                                    const auto row_offset = block.weight_offset + irow * block.tx_count;
                                    for (auto icol{0}; icol < block.tx_count; ++icol) {
                                        dweights[row_offset + icol] += delta_rx * state.signals[block.itx_begin + icol];
                                    }
                                }
                                ipart += block.rx_count - 1;
                                continue;
                            }
                            for (const auto &iconn : std::views::iota(partition.iconn_begin, partition.iconn_end)) {
                                const auto irx = view.connection(iconn).irx;
                                const auto itx = view.connection(iconn).itx;
                                dweights[iconn] += deltas[irx] * state.signals[itx];
                            }
                        }
                        return loss;
                    }
                // Data members
                real_t _learning_rate;
                Optimizer _optimizer;
                int_t _batch_size{1};
                Loss _loss;
                std::vector<NNetState> _states{};
                std::vector<std::vector<real_t>> _deltas{};
                std::vector<std::vector<real_t>> _thread_dweights{};
                std::vector<real_t> _batch_dweights{};
        };
}

