#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class BackpropRule {
        public:
            // Constructor
            explicit BackpropRule(const real_t learning_rate = 0.2) : _learning_rate(learning_rate) {}
            // Learn from targets using the back-propagation learning rule
            template <NNetType Net>
                void learn(Net &net, const output_t<Net> &targets) const {
                    auto view = net.learning_view();
                    std::vector<real_t> deltas(view.neuron_count());
                    std::vector<size_t> number_rx_deltas(view.neuron_count(), 0);
                    index_t itarget{0};
                    // Recursive lambda function to backpropagate errors through the network
                    auto backprop_error = [&] (this auto&& self, const index_t ineuron) -> void {
                        for (const auto iconn : view.in_connections(ineuron)) {
                            const auto& conn = view.connection(iconn);
                            const auto itx = view.neuron_index(conn.tx);
                            deltas[itx] += deltas[ineuron] * conn.weight;
                            if (++number_rx_deltas[itx] == view.out_connections(itx).size()) {
                                deltas[itx] *= conn.tx->activation_derivative(conn.tx->weighted_sum());
                                self(itx);
                            }
                        }
                    };
                    // Compute deltas for output neurons
                    for (index_t ineuron{0}; ineuron < view.neuron_count(); ++ineuron) {
                        if (not view.out_connections(ineuron).empty()) {
                            continue;
                        }
                        const auto& neuron = view.neuron(ineuron);
                        const auto target = targets[itarget++];
                        const auto error = (target - neuron.signal());
                        deltas[ineuron] = error * neuron.activation_derivative(neuron.weighted_sum());
                        backprop_error(ineuron);
                    }
                    // Update weights for all connections
                    for (index_t iconn{0}; iconn < view.connection_count(); ++iconn) {
                         auto& conn = view.connection(iconn);
                         const auto irx = view.neuron_index(conn.rx);
                         conn.weight += _learning_rate * deltas[irx] * conn.tx->signal();
                    }
                }
        private:
            // Data members    
            real_t _learning_rate;
    };
}
