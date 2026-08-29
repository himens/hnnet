#pragma once
#include "hnnet/activation.h"

namespace hNNet {
    //////////////////
    // Neuron class //
    //////////////////
    enum class NeuronType { input, hidden, output, bias };
    class Neuron {
        public:
            // Constructor
            Neuron(const NeuronType type, std::unique_ptr<Activation> activation) : _type(type), _activation(std::move(activation)) {
                if (_activation == nullptr) {
                    throw std::invalid_argument("Neuron::Neuron: invalid activation!");
                }
            }
            // Get neuron signal 
            real_t signal() const {
                return _signal;
            }
            // Get neuron type
            NeuronType type() const {
                return _type;
            }
            // Get weighted sum
            real_t weighted_sum() const {
                return _weighted_sum;
            }
            // Get the activation function
            const Activation*  activation() const {
                return _activation.get();
            }
            // Activate neuron (calculate activation value)
            void activate(const real_t weighted_sum) {
                _weighted_sum = weighted_sum;
                _signal = (*_activation)(_weighted_sum);
            }
            // Reset state before processing a new sample
            void reset() {
                _weighted_sum = 0.0;
                _signal = 0.0;
            }
        private:
            // Data members
            NeuronType _type;
            real_t _weighted_sum{0.0};
            real_t _signal{0.0};
            std::unique_ptr<Activation> _activation{nullptr};
    };
    template <typename T>
        concept NeuronRange = std::ranges::range<T> and std::same_as<std::ranges::range_value_t<T>, Neuron>;
    template <typename T>
        concept NeuronView = std::ranges::view<T> and NeuronRange<T>;
}
