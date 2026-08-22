#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class PerceptronRule {
        public:
            // Constructor
            explicit PerceptronRule(const real_t learning_rate) : _learning_rate(learning_rate) {}
            // Learn from targets using the perceptron learning rule
            template <NNetType Net>
                void learn(Net &net, const output_t<Net> &targets) {
                    auto view = net.view();
                    index_t itarget{0};
                    for (size_t ineuron{0}; ineuron < view.neuron_count(); ++ineuron) {
                        const auto& neuron = view.neuron(ineuron);
                        if (neuron.type() != NeuronType::output) {
                            continue;  // Skip non-output neurons
                        }
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
