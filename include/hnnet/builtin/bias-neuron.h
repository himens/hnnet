#pragma once
#include "backprop-neuron.h"

//////////////////////
// BiasNeuron class //
//////////////////////
class BiasNeuron : public BackpropNeuron {
    public:
        BiasNeuron() : BackpropNeuron() {
            this->add_connection(&_hidden_conn); // trick to avoid marking the neuron as an input neuron
            this->set_signal(1.0);
            _hidden_conn.weight = 0.5;
        }
    private:
        // Learn from error
        //void learn(const hNNet::real_t error) final {}
        // Data members
        SynapticConn _hidden_conn{nullptr, this};
};