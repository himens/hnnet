#pragma once
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Identity : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 1.0_real;
            }
    };
}
