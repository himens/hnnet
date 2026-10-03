#pragma once
#include "hnnet/activation.h"

namespace hnnet::builtin {
    class IdentityActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 1.0_real;
            }
    };
    class LinearActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return slope * x + offset;
            }
            constexpr real_t derivative(const real_t x) const final {
                return slope;
            }
            real_t slope{1.0_real};
            real_t offset{0.0_real};
    };
    class StepActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= threshold ? 1.0_real : 0.0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0_real;
            }
            real_t threshold{0.0_real};
    };
    class BipolarStepActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= threshold ? +1.0_real : -1.0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0_real;
            }
            real_t threshold{0.0_real};
    };
    class PerceptronActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x > threshold ? +1_real : x < -threshold ? -1_real : 0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0_real;
            }
            real_t threshold{0.2_real};
    };
    class SigmoidActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return 1.0_real / (1.0_real + std::exp(-sigma * x));
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return sigma * sig * (1.0_real - sig);
            }
            real_t sigma{1.0_real};
    };
    class BipolarSigmoidActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return 2.0_real / (1.0_real + std::exp(-sigma * x)) - 1.0_real;
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return 0.5_real * sigma * (1.0_real + sig) * (1.0_real - sig);
            }
            real_t sigma{1.0_real};
    };
    class ReLUActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::max(0.0_real, x);
            }
            constexpr real_t derivative(const real_t x) const final {
                return x > 0.0_real ? 1.0_real : 0.0_real;
            }
    };
    class LeakyReLUActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= 0.0_real ? x : 0.01_real * x;
            }
            constexpr real_t derivative(const real_t x) const final {
                return x >= 0.0_real ? 1.0_real : 0.01_real;
            }
    };
    class ELUActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= 0.0_real ? x : 0.01_real * (std::exp(x) - 1.0_real);
            }
            constexpr real_t derivative(const real_t x) const final {
                return x >= 0.0_real ? 1.0_real : 0.01_real * std::exp(x);
            }
    };
    class GaussianActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::exp(-std::pow(x - mean, 2_real) / (2_real * sigma * sigma));
            }
            constexpr real_t derivative(const real_t x) const final {
                return - (x - mean) / (sigma * sigma) * (*this)(x);
            }
            real_t mean{0.0_real};
            real_t sigma{1.0_real};
    };
    class TanhActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::tanh(x);
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto t = (*this)(x);
                return 1.0_real - t * t;
            }
    }; 
    class SoftplusActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::log(1.0_real + std::exp(x));
            }
            constexpr real_t derivative(const real_t x) const final {
                return 1.0_real / (1.0_real + std::exp(-x));
            }
    };
    class SoftsignActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x / (1.0_real + std::abs(x));
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto denom = 1.0_real + std::abs(x);
                return 1.0_real / (denom * denom);
            }
    };
}
