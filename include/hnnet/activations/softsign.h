#pragma once
#include <cmath>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Softsign {
        real_t operator()(const real_t x) const {
            return x / (real_t(1.0) + std::abs(x));
        }
        real_t derivative(const real_t x) const {
            const auto denom = real_t(1.0) + std::abs(x);
            return real_t(1.0) / (denom * denom);
        }
    };
}
