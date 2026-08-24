#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class BackpropRule {
        public:
            // Constructor
            explicit BackpropRule(const real_t learning_rate, const real_t momentum = 0.0) : _learning_rate(learning_rate), _momentum(momentum) {}
            // Learn from targets using the back-propagation learning rule
            template <NNetType Net>
                void learn(Net &net, const output_t<Net> &targets) {
                    // Backpropagate errors through the net
                    auto backprop_error = [&] (this auto&& self, Net::View &view, const index_t ineuron) -> void {
                        for (const auto &iconn : view.in_connections(ineuron)) {
                            const auto itx = view.itx(iconn);
                            _deltas[itx] += _deltas[ineuron] * view.weight(iconn);
                            if (++_number_rx_deltas[itx] == view.out_connections(itx).size()) {
                                const auto &tx = view.neuron(itx);
                                _deltas[itx] *= tx.activation_derivative(tx.weighted_sum());
                                self(view, itx);
                            }
                        }
                    };
                    // Reset internal data
                    auto reset = [&] (const size_t neuron_count, const size_t connection_count) {
                        _number_rx_deltas.clear();
                        _number_rx_deltas.resize(neuron_count);
                        _deltas.clear();
                        _deltas.resize(neuron_count);
                        if (_dweights.empty()) {
                            _dweights.resize(connection_count);
                            std::ranges::fill(_dweights, 0.0);
                        }
                    };
                    auto view = net.view();
                    reset(view.neuron_count(), view.connection_count());
                    index_t itarget{0};
                    // Compute deltas for output neurons and back-propagate through the net
                    for (size_t ineuron{0}; ineuron < view.neuron_count(); ++ineuron) {
                        const auto& neuron = view.neuron(ineuron);
                        if (neuron.type() != NeuronType::output) {
                            continue;  // Skip non-output neurons
                        }
                        const auto target = targets[itarget++];
                        const auto error = (target - neuron.signal());
                        _deltas[ineuron] = error * neuron.activation_derivative(neuron.weighted_sum());
                        backprop_error(view, ineuron);
                    }
                    // Update weights for all connections
                    for (index_t iconn{0}; iconn < view.connection_count(); ++iconn) {
                         const auto irx = view.irx(iconn);
                         const auto &tx = view.neuron(view.itx(iconn));
                         auto &weight = view.weight(iconn);
                         const auto old_weight = weight;
                         weight += (_learning_rate * _deltas[irx] * tx.signal()) + (_momentum * _dweights[iconn]);
                        _dweights[iconn] = (weight - old_weight);  // Store the weight change for momentum
                    }
                }
        private:
            // Data members    
            real_t _learning_rate{0.0};
            real_t _momentum{0.0};
            std::vector<real_t> _deltas{};
            std::vector<real_t> _dweights{};
            std::vector<size_t> _number_rx_deltas{};
    };
}
