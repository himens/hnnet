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
#include <memory>
#include <string>

namespace hNNet {
    using real_t   = std::float64_t;
    using int_t    = std::int64_t;
    using real32_t = std::float32_t;
    using int32_t  = std::int32_t;
    using size_t   = std::size_t;

    template <typename T>
        concept ValueType = std::same_as<T, real_t>   or 
                            std::same_as<T, int_t>    or 
                            std::same_as<T, real32_t> or 
                            std::same_as<T, int32_t>;
    template <size_t Size, ValueType Type = real_t>
        requires (Size > 0)
        using Data = std::array<Type, Size>;
}