#pragma once
#include "hnnet-neuron.h"

namespace Perceptron {
    using namespace hNNet;

    //////////////////
    // Neuron class //
    /////////////////
    class Neuron : public hNNet::Neuron {
        private :
            // Update connection weight according to perceptron learning rule
            void update(SynapticConn *conn, const real_t target) {
            static constexpr real_t rate{1.0};
            conn->weight += rate * target * conn->tx->get_signal();
            }
            // Activation function
            real_t activation(const real_t weighted_sum) const {
                static constexpr real_t threshold{0.2};
                return weighted_sum > threshold ? +1 : weighted_sum < -threshold ? -1 : 0;
            }
    };
}
