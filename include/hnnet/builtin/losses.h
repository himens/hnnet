#pragma once
#include "hnnet/loss.h"

namespace hnnet::builtin {
    ///////////////////  
    // MSELoss class //
    ///////////////////
    class MSELoss {
        public:
            constexpr real_t operator()(const real_t target, const real_t signal) const {
                const auto error = target - signal;
                return 0.5 * error * error;
            }
            constexpr real_t derivative(const real_t target, const real_t signal) const {
                return signal - target;
            }
    };
}
