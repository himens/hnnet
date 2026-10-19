#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Softsign : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x / (1.0_real + std::abs(x));
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto denom = 1.0_real + std::abs(x);
                return 1.0_real / (denom * denom);
            }
    };
}
