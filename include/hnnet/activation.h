#pragma once
#include "hnnet/types.h"

namespace hNNet {
    class Activation {
        public:
            virtual real_t operator()(const real_t x) const = 0;
            virtual real_t derivative(const real_t x) const = 0;
    };
    template <typename T>
        concept ActivationType = std::derived_from<T, Activation>;
}
