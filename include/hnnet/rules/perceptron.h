#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
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
            // Initialize the perceptron rule
            void init(NNet::View view, const TrainingDataType auto &data) {
                const auto weights = view.weights();
                _weights.assign(weights.begin(), weights.end());
                _inputs.clear();
                for (const auto &input : data.inputs) {
                    _inputs.emplace_back(input);
                }
                _targets.clear();
                for (const auto &target : data.targets) {
                    if (std::ssize(target) != view.output_neuron_count()) {
                        throw std::invalid_argument("Perceptron::init: target size does not match the network");
                    }
                    _targets.emplace_back(target);
                }
                _view = view;
            }
            // Learn from a whole epoch of training samples
            bool learn() {
                NNet::State state(_view->neuron_count());
                real_t mean_squared_error{0.0_real};
                for (const auto &[input, target] : std::views::zip(_inputs, _targets)) {
                    update_state(state, input);
                    mean_squared_error += update_weights(state, target);
                }
                constexpr real_t loss_threshold{5e-3_real};
                return mean_squared_error / std::ranges::size(_inputs) < loss_threshold;
            }
            // Apply the learned weights
            void apply(std::vector<real_t> &weights) {
                weights = std::move(_weights);
            }
        private:
            // Update state from input (forward pass over connections)
            void update_state(NNet::State &state, const DataType auto &input) const {
                std::ranges::fill(state.weighted_sums(), 0.0_real);
                state.seed(*_view, input);
                const auto neurons = _view->neurons();
                auto signals = state.signals();
                auto weighted_sums = state.weighted_sums();
                for (const auto &[connection_index, connection] : _view->connections() | std::views::enumerate) {
                    weighted_sums[connection.rx_index] += _weights[connection_index] * signals[connection.tx_index];
                }
                for (const auto &output_neuron_index : _view->output_neurons_indices()) {
                    signals[output_neuron_index] = neurons[output_neuron_index].activate(weighted_sums[output_neuron_index]);
                }
            }
            // Update weights from targets using the perceptron learning rule
            real_t update_weights(NNet::State &state, const DataType auto &targets) {
                real_t squared_error{0.0_real};
                const auto signals = state.signals();
                for (const auto &[target, output_neuron_index] : std::views::zip(targets, _view->output_neurons_indices())) {
                    const auto signal = signals[output_neuron_index];
                    squared_error += _mean_squared_error(target, signal);
                    if (std::abs(target - signal) < 1e-6_real) {
                        continue;  // No update needed if the error is negligible
                    }
                    for (const auto &[connection_index, connection] : _view->connections() | std::views::enumerate) {
                        if (connection.rx_index == output_neuron_index) {
                            _weights[connection_index] += _learning_rate * target * signals[connection.tx_index];
                        }
                    }
                }
                return squared_error;
            }
            // Data members
            real_t _learning_rate;
            loss::MSE _mean_squared_error;
            std::optional<NNet::View> _view{};
            std::vector<std::span<const real_t>> _inputs{};
            std::vector<std::span<const real_t>> _targets{};
            std::vector<real_t> _weights{};
    };
}
