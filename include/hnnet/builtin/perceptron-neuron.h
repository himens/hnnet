#pragma once
#include "hnnet/neuron.h"
#include "hnnet/builtin/activations.h"

namespace hNNet {
    namespace Builtin {
        ////////////////////////////
        // PerceptronNeuron class //
        ////////////////////////////
        class PerceptronNeuron : public Neuron {
            public:
                // Constructor
                PerceptronNeuron() : Neuron(std::make_unique<PerceptronActivation>()) {}
            private:
                // Learn from error
                void learn(const real_t error) final {
                    static constexpr real_t rate{1.0};
                    for (const auto &conn : this->get_in_connections()) {
                        const auto learnt = (std::abs(error) < 1e-6);
                        if (learnt) {
                            return;
                        }
                        const auto target = error + this->get_signal();
                        conn->weight += rate * target * conn->tx->get_signal();
                    }
                }
        };
    }
}