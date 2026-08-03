#pragma once
#include <print>
#include <stdexcept>
#include <ranges>
#include <cstdint>
#include <stdfloat>
#include <array>
#include <vector>
#include <span>
#include <string>
#include <algorithm>
#include <memory>

namespace hNNet {
    using real_t = std::float64_t;
    using int_t = std::int64_t;
    using real_lite_t = std::float32_t;
    using int_lite_t = std::int32_t;

    template <size_t Size>
        using Data = std::array<real_t, Size>;
    template<size_t SizeInput, size_t SizeTarget>
        struct TrainData {
            Data<SizeInput> inputs;
            Data<SizeTarget> targets;
        };
} 