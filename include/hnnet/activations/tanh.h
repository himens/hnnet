#pragma once
#include <cmath>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Tanh {
        real_t operator()(const real_t x) const {
            return std::tanh(x);
        }
        real_t derivative(const real_t x) const {
            const auto t = (*this)(x);
            return real_t(1.0) - t * t;
        }
    };
}
