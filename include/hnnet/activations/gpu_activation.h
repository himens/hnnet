#pragma once
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

namespace hnnet::activation {
    ///////////////////
    // GpuActivation //
    ///////////////////
    // Holds any supported GPU activation by value; compiled by the host and by the GPU backends
    struct GpuActivation {
        // Constructors
        explicit GpuActivation(const BipolarSigmoid activation) : kind(0), bipolar_sigmoid(activation) {}
        explicit GpuActivation(const BipolarStep activation) : kind(1), bipolar_step(activation) {}
        explicit GpuActivation(const ELU activation) : kind(2), elu(activation) {}
        explicit GpuActivation(const Gaussian activation) : kind(3), gaussian(activation) {}
        explicit GpuActivation(const Identity activation) : kind(4), identity(activation) {}
        explicit GpuActivation(const LeakyReLU activation) : kind(5), leaky_relu(activation) {}
        explicit GpuActivation(const Linear activation) : kind(6), linear(activation) {}
        explicit GpuActivation(const Perceptron activation) : kind(7), perceptron(activation) {}
        explicit GpuActivation(const ReLU activation) : kind(8), relu(activation) {}
        explicit GpuActivation(const Sigmoid activation) : kind(9), sigmoid(activation) {}
        explicit GpuActivation(const Softplus activation) : kind(10), softplus(activation) {}
        explicit GpuActivation(const Softsign activation) : kind(11), softsign(activation) {}
        explicit GpuActivation(const Step activation) : kind(12), step(activation) {}
        explicit GpuActivation(const Tanh activation) : kind(13), tanh(activation) {}
        // Data members
        unsigned int kind;
        union {
            BipolarSigmoid bipolar_sigmoid;
            BipolarStep bipolar_step;
            ELU elu;
            Gaussian gaussian;
            Identity identity;
            LeakyReLU leaky_relu;
            Linear linear;
            Perceptron perceptron;
            ReLU relu;
            Sigmoid sigmoid;
            Softplus softplus;
            Softsign softsign;
            Step step;
            Tanh tanh;
        };
        // Compute the activation value
        real_t operator()(const real_t x) const {
            switch (kind) {
                case 0: return bipolar_sigmoid(x);
                case 1: return bipolar_step(x);
                case 2: return elu(x);
                case 3: return gaussian(x);
                case 4: return identity(x);
                case 5: return leaky_relu(x);
                case 6: return linear(x);
                case 7: return perceptron(x);
                case 8: return relu(x);
                case 9: return sigmoid(x);
                case 10: return softplus(x);
                case 11: return softsign(x);
                case 12: return step(x);
                case 13: return tanh(x);
            }
            return real_t(0);
        }
        // Compute the activation derivative
        real_t derivative(const real_t x) const {
            switch (kind) {
                case 0: return bipolar_sigmoid.derivative(x);
                case 1: return bipolar_step.derivative(x);
                case 2: return elu.derivative(x);
                case 3: return gaussian.derivative(x);
                case 4: return identity.derivative(x);
                case 5: return leaky_relu.derivative(x);
                case 6: return linear.derivative(x);
                case 7: return perceptron.derivative(x);
                case 8: return relu.derivative(x);
                case 9: return sigmoid.derivative(x);
                case 10: return softplus.derivative(x);
                case 11: return softsign.derivative(x);
                case 12: return step.derivative(x);
                case 13: return tanh.derivative(x);
            }
            return real_t(0);
        }
    };
}
