#pragma once
#include "hnnet/types.h"

namespace hnnet::loss {
    ///////////////
    // MSE class //
    ///////////////
    class MSE {
        public:
            // Compute the loss
            constexpr real_t operator()(const real_t target, const real_t signal) const {
                const auto error = target - signal;
                return 0.5_real * error * error;
            }
            // Compute the derivative wrt signal
            constexpr real_t derivative(const real_t target, const real_t signal) const {
                return signal - target;
            }
    };
}
