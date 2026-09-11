#pragma once
#include <random>

// Get random generator
std::mt19937& random_generator() {
    static std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    return gen;
}
