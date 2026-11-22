#pragma once
#include "hnnet/activation.h"

namespace hnnet {
    //////////////////
    // Neuron class //
    //////////////////
    enum class NeuronType { input, hidden, output, bias };
    class Neuron {
        public:
            // Constructor
            explicit Neuron(const NeuronType type, const ActivationDescriptor activation) : _type(type), _activation(activation) {}
            // Get neuron type
            NeuronType type() const {
                return _type;
            }
            // Get the activation function
            const ActivationDescriptor& activation() const {
                return _activation;
            }
            // Activate neuron (calculate activation value), returns the computed signal
            real_t activate(const real_t weighted_sum) const {
                return hnnet::activate(_activation, weighted_sum);
            }
        private:
            // Data members
            NeuronType _type;
            ActivationDescriptor _activation;
    };
}
