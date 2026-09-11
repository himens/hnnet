#pragma once
#include <print>
#include <stdexcept>
#include <ranges>
#include <concepts>
#include <stdfloat>
#include <array>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <memory>
#include <unordered_map>
#include <omp.h>
#include "timer.h"

namespace hNNet {
    // Aliases for commonly used types
    using real_t   = std::float64_t;
    using int_t    = std::int64_t;
    using real32_t = std::float32_t;
    using int32_t  = std::int32_t;
    using index_t  = int_t; // indices are just a semantic alias for int_t
    // Constants
    constexpr int_t register_size{4}; // SIMD register size
    // Data type and concepts for fixed-size arrays
    template <typename T>
        concept ValueType = std::same_as<T, real_t>   or 
                            std::same_as<T, int_t>    or 
                            std::same_as<T, real32_t> or 
                            std::same_as<T, int32_t>  or
                            std::same_as<T, index_t>;
    template <ValueType Type, int_t Size>
        requires (Size > 0)
        using Data = std::array<Type, Size>;
    template <typename T>
        concept DataType = requires{ {std::tuple_size_v<T>}; } and std::same_as<T, Data<std::ranges::range_value_t<T>, std::tuple_size_v<T>>>;
    template <typename T>
        concept IndexRange = std::ranges::input_range<T> and std::same_as<std::ranges::range_value_t<T>, index_t>;
}
