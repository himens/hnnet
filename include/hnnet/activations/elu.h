#pragma once
#include <cmath>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct ELU {
        real_t operator()(const real_t x) const {
            return x >= real_t(0.0) ? x : real_t(0.01) * (std::exp(x) - real_t(1.0));
        }
        real_t derivative(const real_t x) const {
            return x >= real_t(0.0) ? real_t(1.0) : real_t(0.01) * std::exp(x);
        }
    };
}
