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
                        for (const auto iconn : view.in_connections(ineuron)) {
                            const auto& conn = view.connection(iconn);
                            const auto itx = view.neuron_index(conn.tx);
                            _deltas[itx] += _deltas[ineuron] * conn.weight;
                            if (++_number_rx_deltas[itx] == view.out_connections(itx).size()) {
                                _deltas[itx] *= conn.tx->activation_derivative(conn.tx->weighted_sum());
                                self(view, itx);
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
                    for (const auto ineuron : view.out_neurons_indices()) {
                        const auto& neuron = view.neuron(ineuron);
                        const auto target = targets[itarget++];
                        const auto error = (target - neuron.signal());
                        _deltas[ineuron] = error * neuron.activation_derivative(neuron.weighted_sum());
                        backprop_error(view, ineuron);
                    }
                    // Update weights for all connections
                    for (index_t iconn{0}; iconn < view.connection_count(); ++iconn) {
                         auto& conn = view.connection(iconn);
                         const auto irx = view.neuron_index(conn.rx);
                         conn.weight += _learning_rate * _deltas[irx] * conn.tx->signal();
                    }
                }
        private:
            // Data members    
            real_t _learning_rate;
            std::vector<real_t> _deltas{};
            std::vector<size_t> _number_rx_deltas{};
    };
}
