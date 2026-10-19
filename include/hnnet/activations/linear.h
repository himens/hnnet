#pragma once
#include "hnnet/activation.h"

namespace hnnet::activation {
    class Linear : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return slope * x + offset;
            }
            constexpr real_t derivative(const real_t x) const final {
                return slope;
            }
            real_t slope{1.0_real};
            real_t offset{0.0_real};
    };
}
