#pragma once
#include "hnnet/neuron.h"
#include "hnnet/builtin/activations.h"

namespace hNNet {
    namespace Builtin {
        //////////////////////////
        // BackpropNeuron class //
        //////////////////////////
        template <ActivationType Activation>
        class BackpropNeuron : public Neuron {
            public:
                // Constructor
                BackpropNeuron() : Neuron(std::make_unique<Activation>()) {}
            private:
                // Learn from error
                void learn(const real_t error) final {
                    const auto derivative = this->get_activation().derivative(this->get_weighted_sum());
                    const auto delta = error * derivative;
                    for (const auto &conn : this->get_in_connections()) {
                        SynapticConn back_conn{.tx = this, .rx = conn->tx, .weight = conn->weight};
                        back_conn.send<BackpropNeuron>(delta, &BackpropNeuron::receive_delta);
                        update(conn, delta);
                    }
                }
                // Receive error from synaptic connection
                void receive_delta(const real_t delta) {
                    _rx_weighted_delta_sum += delta;
                    _number_rx_deltas++;
                    if (_number_rx_deltas == this->get_out_connections().size()) {
                        learn(_rx_weighted_delta_sum);
                        _number_rx_deltas = 0;
                        _rx_weighted_delta_sum = 0.0;
                    }
                }
                // Update synaptic weight
                void update(SynapticConn *conn, const real_t delta) {
                    if ((conn == nullptr) or (conn->tx == nullptr)) {
                        throw std::runtime_error("BackpropNeuron::update: invalid connection!");
                    }
                    static constexpr real_t rate{0.2};
                    conn->weight += rate * delta * conn->tx->get_signal();
                }
                // Data members
                size_t _number_rx_deltas{0};
                real_t _rx_weighted_delta_sum{0.0};
        };
    }
}