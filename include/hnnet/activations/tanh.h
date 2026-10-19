#pragma once
#include <cmath>
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Tanh : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::tanh(x);
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto t = (*this)(x);
                return 1.0_real - t * t;
            }
    };
}
