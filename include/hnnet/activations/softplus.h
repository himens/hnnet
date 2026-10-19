#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Softplus : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::log(1.0_real + std::exp(x));
            }
            constexpr real_t derivative(const real_t x) const final {
                return 1.0_real / (1.0_real + std::exp(-x));
            }
    };
}
