#pragma once
#include "hnnet-activation.h"

namespace hNNet {
    ///////////////////////
    // Linear activation //
    ///////////////////////
    class LinearActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return slope * x + offset;
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return slope;
            }
            // Data members
            real_t slope{1.0};
            real_t offset{0.0};
    };
    /////////////////////
    // Step activation //
    /////////////////////
    class StepActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x >= threshold ? 1.0 : 0.0;
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return 0.0; // Derivative is not defined at x=0, but we can return 0 for practical purposes
            }
            // Data members
            real_t threshold{0.0};
    };
    /////////////////////////////
    // Bipolar step activation //
    /////////////////////////////
    class BipolarStepActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x >= threshold ? +1.0 : -1.0;
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return 0.0; // Derivative is not defined at x=0, but we can return 0 for practical purposes
            }
            // Data members
            real_t threshold{0.0};
    };
    /////////////////////
    // ReLU activation //
    /////////////////////
    class ReLUActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return std::max(offset, x);
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return x > offset ? 1.0 : 0.0;
            }
            // Data members
            real_t offset{0.0};
    };
    ////////////////////////////
    // Perceptron activation //
    ////////////////////////////
    class PerceptronActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x > threshold ? +1 : x < -threshold ? -1 : 0;
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return 0.0; // Derivative is not defined at x=0, but we can return 0 for practical purposes
            }
            // Data members
            real_t threshold{0.0};
    };
    /////////////////////////
    // Gaussian activation //
    /////////////////////////
    class GaussianActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return std::exp(-std::pow(x - mean, 2) / (2 * sigma * sigma));
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return - (x - mean) / (sigma * sigma) * (*this)(x);
            }
            // Data members
            real_t mean{0.0};
            real_t sigma{1.0};
    };
    ////////////////////////
    // Sigmoid activation //
    ////////////////////////
    class SigmoidActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return  1.0 / (1.0 + std::exp(-x));
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return sig * (1.0 - sig);
            }
    };
    /////////////////////
    // Tanh activation //
    /////////////////////
    class TanhActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return std::tanh(x);
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                const auto t = (*this)(x);
                return 1.0 - t * t;
            }
    };
    ///////////////////////////
    // Leaky ReLU activation //
    ///////////////////////////
    class LeakyReLUActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x >= 0.0 ? x : 0.01 * x;
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return x >= 0.0 ? 1.0 : 0.01;
            }
    };
    ////////////////////
    // ELU activation //
    ////////////////////
    class ELUActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x >= 0.0 ? x : 0.01 * (std::exp(x) - 1.0);
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return x >= 0.0 ? 1.0 : 0.01 * std::exp(x);
            }
    };
    //////////////////////
    // Swish activation //
    //////////////////////
    class SwishActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x * SigmoidActivation()(x);
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                const auto sig = SigmoidActivation()(x);
                return sig + x * sig * (1.0 - sig);
            }
    };
    /////////////////////////
    // Softplus activation //
    /////////////////////////
    class SoftplusActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return std::log(1.0 + std::exp(x));
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                return 1.0 / (1.0 + std::exp(-x));
            }
    };
    /////////////////////////
    // Softsign activation //
    /////////////////////////
    class SoftsignActivation : public Activation {
        public:
            // Return activation value
            real_t operator()(const real_t x) const final {
                return x / (1.0 + std::abs(x));
            }
            // Return activation derivative
            real_t derivative(const real_t x) const final {
                const auto denom = 1.0 + std::abs(x);
                return 1.0 / (denom * denom);
            }
    };
}