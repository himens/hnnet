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
        void update(SynapticConn *in_conn, const hNNet::real_t target) final {
            const auto tx_signal = in_conn->tx->get_signal();
            const auto derivative = this->get_activation().derivative(this->get_weighted_sum());
            const auto out_connections = this->get_out_connections();
            if (out_connections.empty()) {
                const auto error = (target - this->get_signal());
                _delta = error * derivative;
            } 
            else {
                auto weighted_deltas = out_connections | std::views::transform([] (const auto &conn) { return conn->weight * static_cast<BackpropNeuron*>(conn->rx)->_delta; });
                const auto weighted_delta = std::ranges::fold_left(weighted_deltas, 0.0, std::plus<>{});
                _delta = weighted_delta * derivative;
            }
            in_conn->weight += tx_signal * _delta;
        }
        // Data members
        hNNet::real_t _delta{0.0};
};