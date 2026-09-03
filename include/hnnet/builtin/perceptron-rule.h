#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class PerceptronRule {
        public:
            // Constructor
            explicit PerceptronRule(const real_t learning_rate) : _learning_rate(learning_rate) {}
            // Learn from targets using the perceptron learning rule
            template <NNetType Net>
                real_t learn(Net &net, const output_t<Net> &targets) {
                    auto view = net.view();
                    if (view.partitions().size() != output_size_v<Net>) {
                        throw std::runtime_error("PerceptronRule::learn: invalid net!");
                    }
                    real_t squared_error{0};
                    for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                        const auto error = (target - view.signal(iout));
                        squared_error += error * error;
                        if (std::abs(error) < 1e-6) {
                            continue;  // No update needed if the error is negligible
                        }
                        // Find the partition corresponding to this output neuron (irx == iout)
                        for (const auto &partition : view.partitions()) {
                            if (partition.irx != iout) {
                                continue;
                            }
                            for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                                const auto itx = view.connection(icon).itx;
                                view.weight(icon) += _learning_rate * target * view.signal(itx);
                            }
                        }
                    }
                    return squared_error;
                }
        private:
            // Data members
            real_t _learning_rate;
    };
}
