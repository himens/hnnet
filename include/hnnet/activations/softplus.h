#pragma once
#include <cmath>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Softplus {
        real_t operator()(const real_t x) const {
            return std::log(real_t(1.0) + std::exp(x));
        }
        real_t derivative(const real_t x) const {
            return real_t(1.0) / (real_t(1.0) + std::exp(-x));
        }
    };
}
