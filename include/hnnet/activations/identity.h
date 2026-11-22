#pragma once
#include "hnnet/types.h"

namespace hnnet::activation {
    struct Identity {
        real_t operator()(const real_t x) const {
            return x;
        }
        real_t derivative(const real_t x) const {
            return real_t(1.0);
        }
    };
}
