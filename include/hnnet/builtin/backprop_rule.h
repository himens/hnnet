#pragma once
#include "hnnet/nnet.h"
#include "hnnet/builtin/dag_net.h"
#include "hnnet/builtin/losses.h"

namespace hnnet::builtin {
    using delta_vector_t = std::vector<real_t>;
    using grad_vector_t = std::vector<real_t>;
    template <typename T>
        concept OptimizerType = requires (T &optimizer, DAGNet::View &view, const grad_vector_t &gradient) {
            optimizer.update(view, gradient);
        };
    ////////////////////
    // Momentum class //
    ////////////////////
    class Momentum {
        public:
            explicit Momentum(const real_t learning_rate, const real_t momentum = 0.0) : _learning_rate(learning_rate), _momentum(momentum) {
                if (_learning_rate < 0.0) {
                    throw std::invalid_argument("Momentum::Momentum: learning_rate must be >= 0");
                }
                if (_momentum < 0.0) {
                    throw std::invalid_argument("Momentum::Momentum: momentum must be >= 0");
                }
                std::println("Momentum::Momentum: learning_rate: {}, momentum: {}", _learning_rate, _momentum);
            }
            void update(DAGNet::View &view, const grad_vector_t &gradient) {
                if (_prev_dweights.empty()) {
                    _prev_dweights.assign(gradient.size(), 0.0);
                }
                for (auto iconn{0}; iconn < std::ssize(gradient); ++iconn) {
                    const auto dweight = - (_learning_rate * gradient[iconn]) + (_momentum * _prev_dweights[iconn]);
                    view.weight(iconn) += dweight;
                    _prev_dweights[iconn] = dweight;
                }
            }
        private:
            real_t _learning_rate{0.0};
            real_t _momentum{0.0};
            grad_vector_t _prev_dweights{};
    };
    /////////////////
    // Rprop class //
    /////////////////
    class Rprop {
        public:
            explicit Rprop(const real_t eta_plus, const real_t eta_minus) : _eta_plus(eta_plus), _eta_minus(eta_minus) {
                if (_eta_plus <= 1.0) {
                    throw std::invalid_argument("Rprop::Rprop: eta_plus must be > 1");
                }
                if (_eta_minus <= 0.0 or _eta_minus >= 1.0) {
                    throw std::invalid_argument("Rprop::Rprop: eta_minus must between 0 and 1");
                }
                std::println("Rprop::Rprop: eta_plus: {}, eta_minus: {}", _eta_plus, _eta_minus);
            }
            void update(DAGNet::View &view, const grad_vector_t &gradient) {
                if (_prev_gradient.empty()) {
                    _prev_gradient.assign(gradient.size(), 0.0);
                }
                if (_deltas.empty()) {
                    _deltas.assign(gradient.size(), 0.1);
                }
                for (auto iconn{0}; iconn < std::ssize(gradient); ++iconn) {
                    constexpr real_t delta_min{1e-6};
                    constexpr real_t delta_max{50.0};
                    auto &delta = _deltas[iconn];
                    auto &prev_grad = _prev_gradient[iconn];
                    const auto &grad = gradient[iconn];
                    const auto sign_prod = prev_grad * grad;
                    if (sign_prod > 0) {
                        delta = utils::math::min(_eta_plus * delta, delta_max);
                        const auto dweight = - utils::math::sign(grad) * delta;
                        view.weight(iconn) += dweight;
                        prev_grad = grad;
                    }
                    else if (sign_prod < 0) {
                        const auto prev_dweight = - utils::math::sign(prev_grad) * delta;
                        delta = utils::math::max(_eta_minus * delta, delta_min);
                        view.weight(iconn) -= prev_dweight;
                        prev_grad = 0.0;
                    }
                    else {
                        const auto dweight = - utils::math::sign(grad) * delta;
                        view.weight(iconn) += dweight;
                        prev_grad = grad;
                    }
                }
            }
        private:
            real_t _eta_plus{0.0};
            real_t _eta_minus{0.0};
            std::vector<real_t> _deltas{};
            grad_vector_t _prev_gradient{};
    };
    ////////////////////////
    // BackpropRule class //
    ////////////////////////
    template <OptimizerType Optimizer, LossType Loss = MSELoss>
        class BackpropRule {
            public:
                // Constructor
                explicit BackpropRule(Optimizer optimizer = Optimizer{}, const int_t batch_size = 1, Loss loss = {}) : _optimizer(optimizer), _batch_size(batch_size), _loss(loss) {
                    if (batch_size == 0) {
                        throw std::invalid_argument("BackpropRule::BackpropRule: batch_size must not be zero");
                    }
                }
                // Learn from a whole epoch of training samples
                real_t learn(DAGNet &net, const DatasetType auto &inputs, const DatasetType auto &targets) {
                    auto view = net.view();
                    const auto neuron_count = view.neuron_count();
                    const auto connection_count = view.connection_count();
                    const auto sample_count = static_cast<index_t>(std::ranges::size(inputs));
                    if (sample_count == 0) {
                        throw std::invalid_argument("BackpropRule::learn: inputs must not be empty");
                    }
                    const auto batch_size = _batch_size < 0 ? sample_count : std::min(_batch_size, sample_count);
                    const auto batch_count = utils::math::ceil(sample_count, batch_size);
                    const auto max_threads = omp_get_max_threads();
                    //std::ranges::shuffle(inputs, utils::random::generator());
                    if (_states.empty()) {
                        _states.assign(max_threads, NNetState{neuron_count});
                        _deltas.assign(max_threads, delta_vector_t(neuron_count, 0.0));
                        _gradients.assign(max_threads, grad_vector_t(connection_count, 0.0));
                        _batch_gradient.assign(connection_count, 0.0);
                    }
                    real_t loss{0.0};
                    for (auto ibatch{0}; ibatch < batch_count; ++ibatch) {
                        const auto batch_begin = ibatch * batch_size;
                        const auto batch_end = std::min(sample_count, batch_begin + batch_size);
                        const auto batch_size = batch_end - batch_begin;
                        const auto parallel = batch_size > 1;
                        const auto thread_count = parallel ? max_threads : index_t{1};
                        for (auto tid{0}; tid < thread_count; ++tid) {
                            std::ranges::fill(_gradients[tid], 0.0);
                        }
                        #pragma omp parallel for if(parallel) reduction(+:loss)
                        for (auto isample = batch_begin; isample < batch_end; ++isample) {
                            const auto tid = omp_get_thread_num();
                            auto &state = _states[tid];
                            net.update(state, inputs[isample]);
                            loss += backward(view, state, targets[isample], _deltas[tid], _gradients[tid]);
                        }
                        if (parallel) {
                            // sequential reduction: sum contribution of each thread
                            std::ranges::fill(_batch_gradient, 0.0);
                            for (auto tid{0}; tid < thread_count; ++tid) {
                                const auto &gradient = _gradients[tid];
                                for (auto iconn{0}; iconn < connection_count; ++iconn) {
                                    _batch_gradient[iconn] += gradient[iconn] / batch_size;
                                }
                            }
                            _optimizer.update(view, _batch_gradient);
                        }
                        else {
                            _optimizer.update(view, _gradients.front());
                        }
                    }
                    return loss / std::ranges::size(inputs);
                }
            private:
                // Compute the error and delta weights contribution of a single sample
                real_t backward(DAGNet::View &view, 
                                NNetState &state, 
                                const DataType auto &targets, 
                                delta_vector_t &deltas, 
                                grad_vector_t &gradient) const {
                    std::ranges::fill(deltas, 0.0);
                    // compute loss and output deltas
                    real_t loss{0.0};
                    for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                        const auto signal = state.signals[iout];
                        loss += _loss(target, signal);
                        deltas[iout] = _loss.derivative(target, signal) * view.neuron(iout).activation()->derivative(state.weighted_sums[iout]);
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
                                if (rx.type() != NeuronType::output) {
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
                            if (rx.type() != NeuronType::output) {
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
                // Data members
                Optimizer _optimizer;
                int_t _batch_size{1};
                Loss _loss;
                std::vector<NNetState> _states{};
                std::vector<delta_vector_t> _deltas{};
                std::vector<grad_vector_t> _gradients{};
                grad_vector_t _batch_gradient{};
        };
}

