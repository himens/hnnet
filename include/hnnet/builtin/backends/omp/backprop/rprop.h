#pragma once
#include "hnnet/builtin/backends/omp/backprop/backend.h"
#include "hnnet/builtin/backends/omp/backprop/optimizers/rprop.h"

namespace hnnet::builtin::backend::omp::backprop {
    /////////////////
    // Rprop class //
    /////////////////
    class Rprop : public Backend<optimizer::Rprop> {
        public:
            using Backend<optimizer::Rprop>::Backend;
            // Initialize the backend
            void init(net::DAGNet &net) {
                Backend::init(net);
                _optimizer.init(net.view().connection_count());
            }
            // Update the weights
            void update_weights() {
                _optimizer.update(_weights, _batch_gradient);
            }
    };
}
