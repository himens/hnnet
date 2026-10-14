#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::builtin::activation {
    class ELU : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= 0.0_real ? x : 0.01_real * (std::exp(x) - 1.0_real);
            }
            constexpr real_t derivative(const real_t x) const final {
                return x >= 0.0_real ? 1.0_real : 0.01_real * std::exp(x);
            }
    };
}