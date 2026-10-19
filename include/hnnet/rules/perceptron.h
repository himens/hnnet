#pragma once
#include <cmath>
#include <print>
#include <ranges>
#include <stdexcept>
#include "hnnet/nnet.h"
#include "hnnet/losses/mse.h"

namespace hnnet::rule {
    ////////////////
    // Rule class //
    ////////////////
    class Perceptron {
        public:
            // Constructor
            explicit Perceptron(const real_t learning_rate) : _learning_rate(learning_rate) {
                if (_learning_rate < 0.0_real) {
                    throw std::invalid_argument("Perceptron: learning_rate must be >= 0");
                }
                std::println("Perceptron: learning rate: {}", learning_rate);
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
                    throw std::runtime_error("Perceptron::learn: invalid net!");
                }
                real_t squared_error{0.0_real};
                const auto connections = view.connections();
                const auto weights = view.weights();
                for (const auto &[target, output_neuron_index] : std::views::zip(targets, view.output_neurons_indices())) {
                    const auto signal = state.signals[output_neuron_index];
                    squared_error += _mean_squared_error(target, signal);
                    if (std::abs(target - signal) < 1e-6_real) {
                        continue;  // No update needed if the error is negligible
                    }
                    // find the partition corresponding to this output neuron
                    for (const auto &partition : view.partitions()) {
                        if (partition.rx_index != output_neuron_index) {
                            continue;
                        }
                        for (const auto &connection_index : std::views::iota(partition.connection_begin, partition.connection_end)) {
                            const auto tx_index = connections[connection_index].tx_index;
                            weights[connection_index] += _learning_rate * target * state.signals[tx_index];
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
