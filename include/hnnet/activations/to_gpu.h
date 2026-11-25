#pragma once
#include <stdexcept>
#include "hnnet/activation.h"
#include "hnnet/activations/gpu_activation.h"

namespace hnnet::activation {
    // Convert a CPU activation to the representation shared by GPU backends
    inline GpuActivation to_gpu(const Activation &activation) {
        if (const auto value = activation.get_if<BipolarSigmoid>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<BipolarStep>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<ELU>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Gaussian>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Identity>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<LeakyReLU>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Linear>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Perceptron>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<ReLU>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Sigmoid>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Softplus>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Softsign>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Step>()) return GpuActivation{*value};
        if (const auto value = activation.get_if<Tanh>()) return GpuActivation{*value};
        throw std::invalid_argument("to_gpu: unsupported or empty activation");
    }
}
