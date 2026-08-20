#pragma once
#include "hnnet/activation.h"

// TODO: 1) Remove all Activation dependency from Neuron and simply add a real_t activate() to NeuronType concept?

namespace hNNet {
    //////////////////
    // Neuron class //
    //////////////////
    class Neuron {
        public:
            // Get neuron signal 
            real_t signal() const {
                return _signal;
            }
            // Get number of received signals
            size_t number_rx_signals() const {
                return _number_rx_signals;
            }
            // Get weighted sum
            real_t weighted_sum() const {
                return _weighted_sum;
            }
            // Activate neuron (calculate activation value)
            void activate() {
                _signal = activation(_weighted_sum);
            }
            // Receive signal from synaptic connection
            void receive_signal(const real_t signal) {
                _weighted_sum += signal;
                _number_rx_signals++;
            }
            // Reset state before processing a new sample
            void reset() {
                _signal = 0.0;
                _number_rx_signals = 0;
                _weighted_sum = 0.0;
            }
        public:
            // Constructor
            Neuron(const Activation &activation) {
                _activation = std::move(activation);
            }
            // Get activation value and derivative
            real_t activation(const real_t x) const {
                return std::visit([&] (const auto& activation) { return activation(x); }, _activation);
            }
            real_t activation_derivative(const real_t x) const {
                return std::visit([&] (const auto& activation) { return activation.derivative(x); }, _activation);
            }
        private:
            // Data members
            size_t _number_rx_signals{0};
            real_t _weighted_sum{0.0};
            real_t _signal{0.0};
            Activation _activation;
    };
    template <typename T>
        concept NeuronType = std::derived_from<T, Neuron>;
    template <typename T>
        concept NeuronRange = std::ranges::range<T> and NeuronType<std::ranges::range_value_t<T>>;
    template <typename T>
        concept NeuronView = std::ranges::view<T> and NeuronRange<T>;
    template <typename T>
        concept NeuronPtr = std::indirectly_readable<std::decay_t<T>> and NeuronType<std::iter_value_t<std::decay_t<T>>>;
    template <typename T>
        concept NeuronPtrRange = std::ranges::range<T> and NeuronPtr<std::ranges::range_value_t<T>>;
}