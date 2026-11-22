#pragma once
#include <cmath>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Gaussian {
        real_t operator()(const real_t x) const {
            return std::exp(-std::pow(x - mean, real_t(2)) / (real_t(2) * sigma * sigma));
        }
        real_t derivative(const real_t x) const {
            return - (x - mean) / (sigma * sigma) * (*this)(x);
        }
        real_t mean{real_t(0.0)};
        real_t sigma{real_t(1.0)};
    };
}
