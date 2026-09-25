#pragma once
#include "hnnet/types.h"

namespace hnnet {
    template <typename T>
        concept LossType = requires (const T &loss, const real_t target, const real_t signal) {
            { loss.operator()(target, signal) } -> std::same_as<real_t>;
            { loss.derivative(target, signal) } -> std::same_as<real_t>;
        };
}
