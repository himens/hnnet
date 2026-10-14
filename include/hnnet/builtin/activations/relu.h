#pragma once
#include <algorithm>
#include "hnnet/activation.h"

namespace hnnet::builtin::activation {
    class ReLU : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::max(0.0_real, x);
            }
            constexpr real_t derivative(const real_t x) const final {
                return x > 0.0_real ? 1.0_real : 0.0_real;
            }
    };
}