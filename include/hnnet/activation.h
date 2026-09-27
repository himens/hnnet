#pragma once
#include "hnnet/types.h"

namespace hnnet {
    class Activation {
        public:
            virtual constexpr real_t operator()(const real_t x) const = 0;
            virtual constexpr real_t derivative(const real_t x) const = 0;
    };
    template <typename T>
        concept ActivationType = std::derived_from<T, Activation>;
}
