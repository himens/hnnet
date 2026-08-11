#pragma once
#include "hnnet-neuron.h"
#include "hnnet-activations.h"

////////////////////////////
// PerceptronNeuron class //
////////////////////////////
class PerceptronNeuron : public hNNet::Neuron {
    public:
        PerceptronNeuron() : hNNet::Neuron(std::make_unique<hNNet::PerceptronActivation>()) {}
    private:
        // Update connection weight according to perceptron learning rule
        void update(SynapticConn *conn, const hNNet::real_t target) final {
            static constexpr hNNet::real_t rate{1.0};
            conn->weight += rate * target * conn->tx->get_signal();
        }
};