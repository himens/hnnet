#pragma once

namespace utils::math {
    // Return ceil of division
    template<typename T>
        requires (std::integral<T> or std::floating_point<T>)
        constexpr T ceil(const T &num, const T &denum) {
            return num / denum + (num % denum != 0);
        }
    // Return sign of value
    template <typename T>
        requires (std::signed_integral<T> or std::floating_point<T>)
        constexpr int sign(const T &val) {
            return (T(0) < val) - (val < T(0));
        }
    // Return min
    template <typename T>
        requires std::totally_ordered_with<T, T>
        constexpr T min(const T &lhs, const T &rhs) {
            return (lhs < rhs ? lhs : rhs);
        }
    // Return max
    template <typename T>
        requires std::totally_ordered_with<T, T>
        constexpr T max(const T &lhs, const T &rhs) {
            return (lhs > rhs ? lhs : rhs);
        }
}

