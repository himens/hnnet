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
            void update(SynapticConn *conn, const hNNet::real_t target) {
            static constexpr hNNet::real_t rate{1.0};
            conn->weight += rate * target * conn->tx->get_signal();
            }
            // Activation function
            hNNet::real_t activation(const hNNet::real_t weighted_sum) const {
                static constexpr hNNet::real_t threshold{0.2};
                return weighted_sum > threshold ? +1 : weighted_sum < -threshold ? -1 : 0;
            }
    };
}
