#pragma once
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Linear {
        real_t operator()(const real_t x) const {
            return slope * x + offset;
        }
        real_t derivative(const real_t x) const {
            return slope;
        }
        real_t slope{real_t(1.0)};
        real_t offset{real_t(0.0)};
    };
}
