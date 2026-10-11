#pragma once
#include "hnnet/types.h"

namespace hnnet::builtin {
    using grad_vector_t = std::vector<real_t>;
    template <typename T, typename Weights, typename Gradient>
        concept OptimizerType = requires (T &optimizer,
                                          Weights &weights,
                                          const Gradient &gradient,
                                          int_t weight_count) {
                                    { optimizer.init(weight_count) } -> std::same_as<void>;
                                    { optimizer.update(weights, gradient) } -> std::same_as<void>;
                                };
}
