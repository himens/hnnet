#pragma once
#include "hnnet/types.h"

namespace hnnet::activation {
    struct LeakyReLU {
        real_t operator()(const real_t x) const {
            return x >= real_t(0.0) ? x : real_t(0.01) * x;
        }
        real_t derivative(const real_t x) const {
            return x >= real_t(0.0) ? real_t(1.0) : real_t(0.01);
        }
    };
}
