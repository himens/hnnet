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
        // Learn from error
        void learn(const hNNet::real_t error) final {
            static constexpr hNNet::real_t rate{1.0};
            for (auto &conn : this->get_in_connections()) {
                const auto learnt = (std::abs(error) < 1e-6);
                if (learnt) {
                    return;
                }
                const auto target = error + this->get_signal();
                conn->weight += rate * target * conn->tx->get_signal();
            }
        }
};