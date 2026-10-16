#pragma once
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Perceptron : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x > threshold ? +1_real : x < -threshold ? -1_real : 0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0_real;
            }
            real_t threshold{0.2_real};
    };
}
