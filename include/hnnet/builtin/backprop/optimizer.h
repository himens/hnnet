#pragma once
#include "hnnet/builtin/dag_net.h"

namespace hnnet::builtin {
    using grad_vector_t = std::vector<real_t>;
    // Optimizer concept
    template <typename T>
        concept OptimizerType = requires (T &optimizer, DAGNet::View &view, const grad_vector_t &gradient) {
            optimizer.update(view, gradient);
        };
}