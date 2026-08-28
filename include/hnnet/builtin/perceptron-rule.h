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
                    if (view.partitions().size() != output_size_v<Net>) {
                        throw std::runtime_error("PerceptronRule::learn: invalid net!");
                    }
                    for (const auto &[i, partition] : view.partitions() | std::views::enumerate) {
                        const auto &rx = view.neuron(partition.irx);
                        if (rx.type() != NeuronType::output) {
                            throw std::runtime_error("PerceptronRule::learn: invalid rx!");
                        }
                        const auto target = targets[i];
                        const auto error = (target - rx.signal());
                        if (std::abs(error) < 1e-6) {
                            continue;  // No update needed if the error is negligible
                        }
                        for (const auto &iconn : std::views::iota(partition.begin, partition.end)) {
                            const auto &tx = view.neuron(view.itx(iconn));
                            view.weight(iconn) += _learning_rate * target * tx.signal();
                        }
                    }
                }
        private:
            // Data members
            real_t _learning_rate;
    };
}
