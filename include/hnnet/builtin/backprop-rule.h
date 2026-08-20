#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class BackpropRule {
        public:
            // Constructor
            explicit BackpropRule(const real_t learning_rate = 0.2) : _learning_rate(learning_rate) {}
            // Learn from targets using the back-propagation learning rule
            template <NeuronType Neuron, DataType InputType, DataType OutputType>
                void learn(NNet<Neuron, InputType, OutputType>& net, const typename NNet<Neuron, InputType, OutputType>::OutputData& targets) {
                    auto view = net.learning_view();
                    std::vector<real_t> deltas(view.neuron_count());
                    std::vector<bool> resolved(view.neuron_count(), false);
                    index_t target_index{0};
                    for (index_t neuron_index{0}; neuron_index < view.neuron_count(); ++neuron_index) {
                        if (not view.outgoing_connections(neuron_index).empty()) {
                            continue;
                        }
                        const auto& neuron = view.neuron(neuron_index);
                        const auto error = targets[target_index++] - neuron.get_signal();
                        deltas[neuron_index] = error * neuron.activation_derivative(neuron.get_weighted_sum());
                        resolved[neuron_index] = true;
                    }
                    const auto delta_for = [&] (this auto&& self, const index_t neuron_index) -> real_t {
                        if (resolved[neuron_index]) {
                            return deltas[neuron_index];
                        }
                        real_t weighted_delta_sum{0.0};
                        for (const auto connection_index : view.outgoing_connections(neuron_index)) {
                            const auto& connection = view.connection(connection_index);
                            weighted_delta_sum += connection.weight * self(view.neuron_index(connection.rx));
                        }
                        const auto& neuron = view.neuron(neuron_index);
                        deltas[neuron_index] = neuron.activation_derivative(neuron.get_weighted_sum()) * weighted_delta_sum;
                        resolved[neuron_index] = true;
                        return deltas[neuron_index];
                    };
                    for (index_t neuron_index{0}; neuron_index < view.neuron_count(); ++neuron_index) {
                        delta_for(neuron_index);
                    }
                    for (index_t connection_index{0}; connection_index < view.connection_count(); ++connection_index) {
                        auto& connection = view.connection(connection_index);
                        const auto receiver_index = view.neuron_index(connection.rx);
                        connection.weight += _learning_rate * deltas[receiver_index] * connection.tx->get_signal();
                    }
                }
        private:
            // Data members    
            real_t _learning_rate;
    };
}
