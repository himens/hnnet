#pragma once
#include <algorithm>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct ReLU {
        real_t operator()(const real_t x) const {
            return std::max(real_t(0.0), x);
        }
        real_t derivative(const real_t x) const {
            return x > real_t(0.0) ? real_t(1.0) : real_t(0.0);
        }
    };
}
