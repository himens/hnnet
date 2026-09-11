#pragma once
#include "hnnet/loss.h"

namespace hNNet::Builtin {
    ///////////////////  
    // MSELoss class //
    ///////////////////
    class MSELoss {
        public:
            real_t operator()(const real_t target, const real_t signal) const {
                const auto error = target - signal;
                return error * error;
            }
            real_t derivative(const real_t target, const real_t signal) const {
                return target - signal;
            }
    };
}
