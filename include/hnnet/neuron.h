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
            explicit Neuron(const NeuronType type, std::unique_ptr<Activation> activation) : _type(type), _activation(std::move(activation)) {
                if (_activation == nullptr) {
                    throw std::invalid_argument("Neuron::Neuron: invalid activation!");
                }
            }
            // Get neuron type
            NeuronType type() const {
                return _type;
            }
            // Get the activation function
            const Activation*  activation() const {
                return _activation.get();
            }
            // Activate neuron (calculate activation value), returns the computed signal
            real_t activate(const real_t weighted_sum) const {
                return (*_activation)(weighted_sum);
            }
        private:
            // Data members
            NeuronType _type;
            std::unique_ptr<Activation> _activation{nullptr};
    };
}
