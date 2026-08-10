#pragma once
#include "hnnet-neuron.h"
#include "hnnet-activations.h"

namespace Perceptron {
    using namespace hNNet;

    //////////////////
    // Neuron class //
    /////////////////
    class Neuron : public hNNet::Neuron {
        private :
            // Update connection weight according to perceptron learning rule
            void update(SynapticConn *conn, const real_t target) final {
                static constexpr real_t rate{1.0};
                conn->weight += rate * target * conn->tx->get_signal();
            }
            // Activation function
            real_t activation(const real_t weighted_sum) const final {
                return perceptron_activation(weighted_sum);
            }
    };
}
