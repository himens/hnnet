#pragma once
#include <random>
#include <ranges>

namespace utils::random {
    // Get random generator
    std::mt19937& generator() {
        static std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        return gen;
    }
    // Generate random numbers
    template <typename T>
        requires(std::integral<T> or std::floating_point<T>)
        std::vector<T> generate(const size_t size, const T min, const T max) {
            std::vector<T> values(size);
            if constexpr(std::integral<T>) {
                std::uniform_int_distribution<T> distrib(min, max);
                std::ranges::generate(values, [&]() { return distrib(generator()); });
            }
            else if constexpr(std::floating_point<T>) {
                std::uniform_real_distribution<T> distrib(min, max);
                std::ranges::generate(values, [&]() { return distrib(generator()); });
            }
            return values;
        }
}
