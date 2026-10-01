#pragma once

namespace utils::math {
    // Return ceil of division
    template<typename T>
        requires (std::integral<T> or std::floating_point<T>)
        constexpr T ceil(T num, T denum) {
            return num / denum + (num % denum != 0);
        }
    // Return sign of value
    template <typename T>
        requires (std::signed_integral<T> or std::floating_point<T>)
        constexpr int sign(T val) {
            return (T(0) < val) - (val < T(0));
        }
}

