#pragma once
#include <cmath>
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Sigmoid {
        real_t operator()(const real_t x) const {
            return real_t(1.0) / (real_t(1.0) + std::exp(-sigma * x));
        }
        real_t derivative(const real_t x) const {
            const auto sig = (*this)(x);
            return sigma * sig * (real_t(1.0) - sig);
        }
        real_t sigma{real_t(1.0)};
    };
}
