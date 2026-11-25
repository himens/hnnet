#pragma once
#include "hnnet/types.h"

namespace hnnet::activation {
    struct BipolarStep {
        real_t operator()(const real_t x) const {
            return x >= threshold ? +real_t(1.0) : -real_t(1.0);
        }
        real_t derivative(const real_t x) const {
            return real_t(0.0);
        }
        real_t threshold{real_t(0.0)};
    };
}
