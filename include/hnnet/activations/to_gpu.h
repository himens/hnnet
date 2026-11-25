#pragma once
#include <stdexcept>
#include "hnnet/activation.h"
#include "hnnet/activations/gpu_activation.h"

namespace hnnet {
    // Convert a CPU activation to the representation shared by GPU backends
    inline GpuActivation to_gpu(const Activation &activation) {
        if (const auto value = activation.get_if<activation::BipolarSigmoid>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::BipolarStep>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::ELU>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Gaussian>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Identity>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::LeakyReLU>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Linear>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Perceptron>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::ReLU>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Sigmoid>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Softplus>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Softsign>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Step>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<activation::Tanh>()) return GpuActivation{*value};
        throw std::invalid_argument("to_gpu: unsupported or empty activation");
    }
}
