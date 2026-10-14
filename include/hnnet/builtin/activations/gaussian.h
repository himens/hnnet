#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::builtin::activation {
    class Gaussian : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::exp(-std::pow(x - mean, 2_real) / (2_real * sigma * sigma));
            }
            constexpr real_t derivative(const real_t x) const final {
                return - (x - mean) / (sigma * sigma) * (*this)(x);
            }
            real_t mean{0.0_real};
            real_t sigma{1.0_real};
    };
}