#pragma once
#include "hnnet/nnet.h"
#include "hnnet/builtin/losses.h"

namespace hNNet::Builtin {
    //////////////////////////
    // PerceptronRule class //
    //////////////////////////
    class PerceptronRule {
            public:
            // Constructor
            explicit PerceptronRule(const real_t learning_rate) : _learning_rate(learning_rate) {
                    if (_learning_rate < 0.0) {
                        throw std::invalid_argument("PerceptronRule::PerceptronRule: learning_rate must be >= 0");
                    }
                    std::println("PerceptronRule::PerceptronRule: learning rate: {}", learning_rate);
                }
                // Learn from a whole epoch of training samples (online: one immediate update per sample)
                template <NNetType Net>
                    real_t learn(Net &net, const std::span<typename Net::TrainingData> samples) {
                        NNetState state(net.view().neuron_count());
                        real_t mean_squared_error{0.0};
                        for (const auto &sample : samples) {
                            net.update(state, sample.inputs);
                            mean_squared_error += update_weights(net, state, sample.targets);
                        }
                        return mean_squared_error / samples.size();
                    }
            private:
                // Update weights from targets using the perceptron learning rule
                template <NNetType Net>
                    real_t update_weights(Net &net, NNetState &state, const output_t<Net> &targets) {
                        auto view = net.view();
                        if (view.partitions().size() != output_size_v<Net>) {
                            throw std::runtime_error("PerceptronRule::learn: invalid net!");
                        }
                        real_t squared_error{0};
                        for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                            const auto signal = state.signals[iout];
                            squared_error += _mean_squared_error(target, signal);
                            if (std::abs(target - signal) < 1e-6) {
                                continue;  // No update needed if the error is negligible
                            }
                            // Find the partition corresponding to this output neuron (irx == iout)
                            for (const auto &partition : view.partitions()) {
                                if (partition.irx != iout) {
                                    continue;
                                }
                                for (const auto &iconn : std::views::iota(partition.iconn_begin, partition.iconn_end)) {
                                    const auto itx = view.connection(iconn).itx;
                                    view.weight(iconn) += _learning_rate * target * state.signals[itx];
                                }
                            }
                        }
                        return squared_error;
                    }
                // Data members
                real_t _learning_rate;
                MSELoss _mean_squared_error;
        };
}
