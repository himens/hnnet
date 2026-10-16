#pragma once
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Step : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= threshold ? 1.0_real : 0.0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0_real;
            }
            real_t threshold{0.0_real};
    };
}
