#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::activation {
    class BipolarSigmoid : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return 2.0_real / (1.0_real + std::exp(-sigma * x)) - 1.0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return 0.5_real * sigma * (1.0_real + sig) * (1.0_real - sig);
            }
            real_t sigma{1.0_real};
    };
}
