#pragma once
#include "hnnet/builtin/backends/omp/backprop/backend.h"
#include "hnnet/builtin/backends/omp/backprop/optimizers/momentum.h"

namespace hnnet::builtin::backend::omp::backprop {
    ////////////////////
    // Momentum class //
    ////////////////////
    // OpenMP backprop backend using the Momentum optimizer
    class Momentum : public Backend {
        public:
            // Constructor
            explicit Momentum(const real_t learning_rate, const real_t momentum = 0.0_real) : _optimizer(learning_rate, momentum) {}
            // Initialize the backend
            void init(net::DAGNet &net) {
                Backend::init(net);
                _optimizer.init(net.view().connection_count());
            }
            // Update the weights
            void update_weights() {
                _optimizer.update(_weights, _batch_gradient);
            }
        private:
            // Data members
            optimizer::Momentum _optimizer;
    };
}
