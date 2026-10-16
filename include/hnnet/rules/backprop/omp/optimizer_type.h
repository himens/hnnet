#pragma once
#include <concepts>
#include <span>
#include "hnnet/types.h"

namespace hnnet::backprop::omp {
    template <typename T>
        concept OptimizerType = requires(T &optimizer,
                                         int_t weight_count,
                                         std::span<real_t> weights,
                                         std::span<const real_t> gradient) {
                                    { optimizer.init(weight_count) } -> std::same_as<void>;
                                    { optimizer.update(weights, gradient) } -> std::same_as<void>;
                                };
}
