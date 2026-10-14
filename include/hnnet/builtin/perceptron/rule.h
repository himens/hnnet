#pragma once
#include <cmath>
#include <print>
#include <ranges>
#include <stdexcept>
#include "hnnet/nnet.h"
#include "hnnet/builtin/losses/mse.h"

namespace hnnet::builtin::perceptron {
    ////////////////
    // Rule class //
    ////////////////
    class Rule {
        public:
            // Constructor
            explicit Rule(const real_t learning_rate) : _learning_rate(learning_rate) {
                if (_learning_rate < 0.0_real) {
                    throw std::invalid_argument("perceptron::Rule: learning_rate must be >= 0");
                }
                std::println("perceptron::Rule: learning rate: {}", learning_rate);
            }
            // Initialize the perceptron rule (if needed)
            void init(NNet &) {}
            // Learn from a whole epoch of training samples (online: one immediate update per sample)
            bool learn(NNet &net, const DatasetType auto &inputs, const DatasetType auto &targets) {
                auto view = net.view();
                NNet::State state(view.neuron_count());
                real_t mean_squared_error{0.0_real};
                for (const auto &[input, target] : std::views::zip(inputs, targets)) {
                    net.update(state, input);
                    mean_squared_error += update_weights(view, state, target);
                }
                constexpr real_t loss_threshold{5e-3_real};
                return mean_squared_error / std::ranges::size(inputs) < loss_threshold;
            }
        private:
            // Update weights from targets using the perceptron learning rule
            real_t update_weights(NNet::View &view, NNet::State &state, const DataType auto &targets) {
                if (view.partitions().size() != std::ranges::size(targets)) {
                    throw std::runtime_error("perceptron::Rule::learn: invalid net!");
                }
                real_t squared_error{0.0_real};
                const auto connections = view.connections();
                const auto weights = view.weights();
                for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                    const auto signal = state.signals[iout];
                    squared_error += _mean_squared_error(target, signal);
                    if (std::abs(target - signal) < 1e-6_real) {
                        continue;  // No update needed if the error is negligible
                    }
                    // find the partition corresponding to this output neuron (irx == iout)
                    for (const auto &partition : view.partitions()) {
                        if (partition.irx != iout) {
                            continue;
                        }
                        for (const auto &iconn : std::views::iota(partition.conn_begin, partition.conn_end)) {
                            const auto itx = connections[iconn].itx;
                            weights[iconn] += _learning_rate * target * state.signals[itx];
                        }
                    }
                }
                return squared_error;
            }
            // Data members
            real_t _learning_rate;
            loss::MSE _mean_squared_error;
    };
}