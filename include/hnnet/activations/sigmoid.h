#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Sigmoid : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return 1.0_real / (1.0_real + std::exp(-sigma * x));
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return sigma * sig * (1.0_real - sig);
            }
            real_t sigma{1.0_real};
    };
}
