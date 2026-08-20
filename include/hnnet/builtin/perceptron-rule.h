#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class PerceptronRule {
        public:
            // Constructor
            explicit PerceptronRule(const real_t learning_rate = 1.0) : _learning_rate(learning_rate) {}
            // Learn from targets using the perceptron learning rule
            template <NeuronType Neuron, DataType Input, DataType Output>
                void learn(NNet<Neuron, Input, Output>& net, const typename NNet<Neuron, Input, Output>::OutputData& targets) const {
                    auto view = net.learning_view();
                    index_t itarget{0};
                    for (index_t ineuron{0}; ineuron < view.neuron_count(); ++ineuron) {
                        if (not view.out_connections(ineuron).empty()) {
                            continue;
                        }
                        const auto& neuron = view.neuron(ineuron);
                        const auto target = targets[itarget++];
                        const auto error = (target - neuron.signal());
                        if (std::abs(error) < 1e-6) {
                            continue;  // No update needed if the error is negligible
                        }
                        for (const auto &iconn : view.in_connections(ineuron)) {
                            auto& conn = view.connection(iconn);
                            conn.weight += _learning_rate * target * conn.tx->signal();
                        }
                    }
                }
        private:
            // Data members
            real_t _learning_rate;
    };
}
