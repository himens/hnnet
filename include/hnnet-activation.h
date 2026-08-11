#pragma once
#include "hnnet.h"

namespace hNNet {
    //////////////////////
    // Activation class //
    //////////////////////
    class Activation {
        public:
            // Destructor
            virtual ~Activation() = default;
            // Return activation value
            virtual real_t operator()(const real_t x) const = 0;
            // Return activation derivative
            virtual real_t derivative(const real_t x) const = 0;
            // Return activation value (alias for operator())
            real_t value(const real_t x) const {
                return (*this)(x);
            }
    };
}