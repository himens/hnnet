#pragma once
#include "hnnet/builtin/backends/omp/backprop/backend.h"
#include "hnnet/builtin/backends/omp/backprop/optimizers/rprop.h"

namespace hnnet::builtin::backend::omp::backprop {
    /////////////////
    // Rprop class //
    /////////////////
    // OpenMP backprop backend using the Rprop optimizer
    class Rprop : public Backend {
        public:
            // Constructor
            explicit Rprop(const real_t eta_plus, const real_t eta_minus)
                : _optimizer(eta_plus, eta_minus) {}
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
            optimizer::Rprop _optimizer;
    };
}
