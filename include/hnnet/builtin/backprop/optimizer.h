#pragma once
#include "hnnet/builtin/dag_net.h"

namespace hnnet::builtin {
    using grad_vector_t = std::vector<real_t>;
    template <typename T>
        concept OptimizerType = requires (T &optimizer, 
                                          DAGNet::View &view, 
                                          const grad_vector_t &gradient,
                                          int_t weight_count) {
                                    { optimizer.init(weight_count) } -> std::same_as<void>;
                                    { optimizer.update(view, gradient) } -> std::same_as<void>;
                                };
}
