#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class BackpropRule {
        public:
            // Constructor
            explicit BackpropRule(const real_t learning_rate) : _learning_rate(learning_rate) {}
            // Learn from targets using the back-propagation learning rule
            template <NNetType Net>
                void learn(Net &net, const output_t<Net> &targets) {
                    // Backpropagate errors through the net
                    auto backprop_error = [&] (this auto&& self, Net::View &view, const index_t ineuron) -> void {
                        for (const auto &iconn : view.in_connections(ineuron)) {
                            const auto& conn = view.connection(iconn);
                            _deltas[conn.itx] += _deltas[ineuron] * conn.weight;
                            if (++_number_rx_deltas[conn.itx] == view.out_connections(conn.itx).size()) {
                                const auto &tx = view.neuron(conn.itx);
                                _deltas[conn.itx] *= tx.activation_derivative(tx.weighted_sum());
                                self(view, conn.itx);
                            }
                        }
                    };
                    // Reset internal data
                    auto reset = [&] (const size_t neuron_count) {
                        _number_rx_deltas.clear();
                        _number_rx_deltas.resize(neuron_count);
                        _deltas.clear();
                        _deltas.resize(neuron_count);

                    };
                    auto view = net.view();
                    reset(view.neuron_count());
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
                         auto& conn = view.connection(iconn);
                         const auto &tx = view.neuron(conn.itx);
                         conn.weight += _learning_rate * _deltas[conn.irx] * tx.signal();
                    }
                }
        private:
            // Data members    
            real_t _learning_rate;
            std::vector<real_t> _deltas{};
            std::vector<size_t> _number_rx_deltas{};
    };
}
