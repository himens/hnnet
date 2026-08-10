#pragma once
#include "hnnet.h"

namespace hNNet {
    real_t gaussian_activation(const real_t x) {
        return std::exp(-std::pow(x, 2));
    }
    real_t linear_activation(const real_t x) {
        return x;
    }
    real_t relu_activation(const real_t x) {
        return std::max(real_t{0.0}, x);
    }
    real_t sigmoid_activation(const real_t x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    real_t perceptron_activation(const real_t x, const real_t thr = 0.2) {
        return x > thr ? +1 : x < -thr ? -1 : 0;
    }
    real_t tanh_activation(const real_t x) {
        return std::tanh(x);
    }
    real_t step_activation(const real_t x) {
        return x >= 0.0 ? +1 : -1;
    }
    real_t swish_activation(const real_t x) {
        return x * sigmoid_activation(x);
    }
    real_t softplus_activation(const real_t x) {
        return std::log(1.0 + std::exp(x));
    }
    real_t softsign_activation(const real_t x) {
        return x / (1.0 + std::abs(x));
    }
}