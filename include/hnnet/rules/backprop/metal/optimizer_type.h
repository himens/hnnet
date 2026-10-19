#pragma once
#include <concepts>
#include <cstdint>
#include "runtime.h"

namespace hnnet::backprop::metal {
    template <typename T>
        concept OptimizerType = requires(T &optimizer,
                                         const detail::Context &context,
                                         std::uint32_t weight_count,
                                         id<MTLCommandBuffer> command,
                                         id<MTLBuffer> weights,
                                         id<MTLBuffer> gradient) {
                                     { optimizer.init(context, weight_count) } -> std::same_as<void>;
                                     { optimizer.update(command, weights, gradient, weight_count) } -> std::same_as<void>;
                                 };
}