#pragma once
#include <concepts>
#include <cstdint>
#include <vector>
#include <ranges>
#define HAS_STDFLOAT __has_include(<stdfloat>)
#if HAS_STDFLOAT
#include <stdfloat>
#endif

namespace hnnet {
    // Aliases for commonly used types
#if HAS_STDFLOAT
    using real_t = std::float32_t;
#else
    using real_t = float;
#endif
    using int_t = std::int64_t;
    using index_t = int_t; // indices are just a semantic alias for int_t
    using dataset_vector_t = std::vector<std::vector<real_t>>;
    constexpr real_t operator""_real(long double value) {
        return static_cast<real_t>(value);
    }
    constexpr real_t operator""_real(unsigned long long value) {
        return static_cast<real_t>(value);
    }
    // Constants
    constexpr int_t simd_register_size{4};
    // Concepts
    template <typename T>
        concept DataType = std::ranges::contiguous_range<T> and std::same_as<std::ranges::range_value_t<T>, real_t>;
    template <typename T>
        concept DatasetType = std::ranges::contiguous_range<const T> and DataType<std::ranges::range_reference_t<const T>>;
    template <typename T>
        concept IndexRange = std::ranges::input_range<T> and std::same_as<std::ranges::range_value_t<T>, index_t>;
    // Data types
    template <DatasetType Inputs, DatasetType Targets = dataset_vector_t>
        struct TrainingData {
            Inputs inputs;
            Targets targets{};
        };
    template <typename T>
        concept TrainingDataType = requires(const T &data) {
                                       requires DatasetType<decltype(data.inputs)>;
                                       requires DatasetType<decltype(data.targets)>;
                                       std::ranges::size(data.inputs);
                                       std::ranges::size(data.targets);
                                   };
}
