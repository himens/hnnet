#pragma once
#include "hnnet-neuron.h"
#include "hnnet-activations.h"

//////////////////////////
// BackpropNeuron class //
//////////////////////////
class BackpropNeuron : public hNNet::Neuron {
    public:
        BackpropNeuron() : hNNet::Neuron(std::make_unique<hNNet::SigmoidActivation>()) {}
    private:
        // Update connection weight according to backprop learning rule
        void update(SynapticConn *conn, const hNNet::real_t target) final {
        }
};