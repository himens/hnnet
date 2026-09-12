#pragma once
#include <print>
#include <stdexcept>
#include <ranges>
#include <concepts>
#include <stdfloat>
#include <cmath>
#include <numeric>
#include <array>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <unordered_map>
#include <omp.h>
#include "timer.h"
#include "random.h"

namespace hNNet {
    // Aliases for commonly used types
    using real_t  = std::float64_t;
    using int_t   = std::int64_t;
    using index_t = int_t; // indices are just a semantic alias for int_t
    using Dataset = std::vector<std::vector<real_t>>;
    // Constants
    constexpr int_t register_size{4}; // SIMD register size
    // Concepts
    template <typename T>
        concept DataType = std::ranges::contiguous_range<T> and std::same_as<std::ranges::range_value_t<T>, real_t>;
    template <typename T>
        concept DatasetType = std::ranges::contiguous_range<const T> and DataType<std::ranges::range_reference_t<const T>>;
    template <typename T>
        concept IndexRange = std::ranges::input_range<T> and std::same_as<std::ranges::range_value_t<T>, index_t>;
}
