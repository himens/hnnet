#pragma once
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Perceptron {
        real_t operator()(const real_t x) const {
            return x > threshold ? +real_t(1) : x < -threshold ? -real_t(1) : real_t(0);
        }
        real_t derivative(const real_t x) const {
            return real_t(0.0);
        }
        real_t threshold{real_t(0.2)};
    };
}
