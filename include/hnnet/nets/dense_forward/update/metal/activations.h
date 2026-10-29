#pragma once
#include <metal_stdlib>

namespace hnnet {
namespace dense_forward {
namespace update {
namespace metal {
namespace activation {
    inline float identity(const float x) {
        return x;
    }

    inline float identity_derivative(const float) {
        return 1.0f;
    }

    inline float sigmoid(const float x, const float sigma = 1.0f) {
        return 1.0f / (1.0f + ::metal::exp(-sigma * x));
    }

    inline float sigmoid_derivative(const float x, const float sigma = 1.0f) {
        const auto signal = sigmoid(x, sigma);
        return sigma * signal * (1.0f - signal);
    }

    inline float relu(const float x) {
        return x > 0.0f ? x : 0.0f;
    }

    inline float relu_derivative(const float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
}
}
}
}
}