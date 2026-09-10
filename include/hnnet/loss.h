#pragma once
#include "hnnet/types.h"

namespace hNNet {
    template <typename T>
        concept LossType = requires (const T &loss, const real_t target, const real_t signal) {
            { loss.value(target, signal) } -> std::same_as<real_t>;
            { loss.derivative(target, signal) } -> std::same_as<real_t>;
        };
}
