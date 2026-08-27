#pragma once
#include <print>
#include <stdexcept>
#include <ranges>
#include <concepts>
#include <type_traits>
#include <cstdint>
#include <stdfloat>
#include <array>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <random>
#include <variant>
#include <omp.h>

namespace hNNet {
    using real_t   = std::float64_t;
    using int_t    = std::int64_t;
    using real32_t = std::float32_t;
    using int32_t  = std::int32_t;
    using size_t   = std::size_t;
    using index_t  = std::uint32_t;

    template <typename T>
        concept ValueType = std::same_as<T, real_t>   or 
                            std::same_as<T, int_t>    or 
                            std::same_as<T, real32_t> or 
                            std::same_as<T, int32_t>;
    template <ValueType Type, size_t Size>
        requires (Size > 0)
        using Data = std::array<Type, Size>;
    template <typename T>
        concept DataType = requires{ {std::tuple_size_v<T>}; } and std::same_as<T, Data<std::ranges::range_value_t<T>, std::tuple_size_v<T>>>;
    template <typename T>
        concept IndexRange = std::ranges::range<T> and std::integral<std::ranges::range_value_t<T>>;
}