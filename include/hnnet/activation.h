#pragma once
#include <concepts>
#include "hnnet/activations/bipolar_sigmoid.h"
#include "hnnet/activations/bipolar_step.h"
#include "hnnet/activations/elu.h"
#include "hnnet/activations/gaussian.h"
#include "hnnet/activations/identity.h"
#include "hnnet/activations/leaky_relu.h"
#include "hnnet/activations/linear.h"
#include "hnnet/activations/perceptron.h"
#include "hnnet/activations/relu.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/activations/softplus.h"
#include "hnnet/activations/softsign.h"
#include "hnnet/activations/step.h"
#include "hnnet/activations/tanh.h"

namespace hnnet {
    /////////////////////////////////
    // ActivationDescriptor struct //
    /////////////////////////////////
    // Holds any activation by value; this file is compiled by the CPU and by the GPU backends
    struct ActivationDescriptor {
        // Constructors
        explicit ActivationDescriptor(const activation::BipolarSigmoid activation) : kind(0), bipolar_sigmoid(activation) {}
        explicit ActivationDescriptor(const activation::BipolarStep activation) : kind(1), bipolar_step(activation) {}
        explicit ActivationDescriptor(const activation::ELU activation) : kind(2), elu(activation) {}
        explicit ActivationDescriptor(const activation::Gaussian activation) : kind(3), gaussian(activation) {}
        explicit ActivationDescriptor(const activation::Identity activation) : kind(4), identity(activation) {}
        explicit ActivationDescriptor(const activation::LeakyReLU activation) : kind(5), leaky_relu(activation) {}
        explicit ActivationDescriptor(const activation::Linear activation) : kind(6), linear(activation) {}
        explicit ActivationDescriptor(const activation::Perceptron activation) : kind(7), perceptron(activation) {}
        explicit ActivationDescriptor(const activation::ReLU activation) : kind(8), relu(activation) {}
        explicit ActivationDescriptor(const activation::Sigmoid activation) : kind(9), sigmoid(activation) {}
        explicit ActivationDescriptor(const activation::Softplus activation) : kind(10), softplus(activation) {}
        explicit ActivationDescriptor(const activation::Softsign activation) : kind(11), softsign(activation) {}
        explicit ActivationDescriptor(const activation::Step activation) : kind(12), step(activation) {}
        explicit ActivationDescriptor(const activation::Tanh activation) : kind(13), tanh(activation) {}
        // Data members
        unsigned int kind;
        union {
            activation::BipolarSigmoid bipolar_sigmoid;
            activation::BipolarStep bipolar_step;
            activation::ELU elu;
            activation::Gaussian gaussian;
            activation::Identity identity;
            activation::LeakyReLU leaky_relu;
            activation::Linear linear;
            activation::Perceptron perceptron;
            activation::ReLU relu;
            activation::Sigmoid sigmoid;
            activation::Softplus softplus;
            activation::Softsign softsign;
            activation::Step step;
            activation::Tanh tanh;
        };
    };
    // Compute the activation value
    inline real_t activate(const ActivationDescriptor descriptor, const real_t x) {
        switch (descriptor.kind) {
            case 0: return descriptor.bipolar_sigmoid(x);
            case 1: return descriptor.bipolar_step(x);
            case 2: return descriptor.elu(x);
            case 3: return descriptor.gaussian(x);
            case 4: return descriptor.identity(x);
            case 5: return descriptor.leaky_relu(x);
            case 6: return descriptor.linear(x);
            case 7: return descriptor.perceptron(x);
            case 8: return descriptor.relu(x);
            case 9: return descriptor.sigmoid(x);
            case 10: return descriptor.softplus(x);
            case 11: return descriptor.softsign(x);
            case 12: return descriptor.step(x);
            case 13: return descriptor.tanh(x);
        }
        return real_t(0);
    }
    // Compute the activation derivative
    inline real_t derivative(const ActivationDescriptor descriptor, const real_t x) {
        switch (descriptor.kind) {
            case 0: return descriptor.bipolar_sigmoid.derivative(x);
            case 1: return descriptor.bipolar_step.derivative(x);
            case 2: return descriptor.elu.derivative(x);
            case 3: return descriptor.gaussian.derivative(x);
            case 4: return descriptor.identity.derivative(x);
            case 5: return descriptor.leaky_relu.derivative(x);
            case 6: return descriptor.linear.derivative(x);
            case 7: return descriptor.perceptron.derivative(x);
            case 8: return descriptor.relu.derivative(x);
            case 9: return descriptor.sigmoid.derivative(x);
            case 10: return descriptor.softplus.derivative(x);
            case 11: return descriptor.softsign.derivative(x);
            case 12: return descriptor.step.derivative(x);
            case 13: return descriptor.tanh.derivative(x);
        }
        return real_t(0);
    }
#ifndef __METAL_VERSION__
    template <typename T>
        concept ActivationType = std::constructible_from<ActivationDescriptor, T>;
#endif
}
