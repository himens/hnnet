#pragma once
#include "hnnet/activation.h"

namespace hnnet::builtin {
    class IdentityActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 1.0;
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
            real_t slope{1.0};
            real_t offset{0.0};
    };
    class StepActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= threshold ? 1.0 : 0.0;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0;
            }
            real_t threshold{0.0};
    };
    class BipolarStepActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= threshold ? +1.0 : -1.0;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0;
            }
            real_t threshold{0.0};
    };
    class PerceptronActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x > threshold ? +1 : x < -threshold ? -1 : 0;
            }
            constexpr real_t derivative(const real_t x) const final {
                return 0.0;
            }
            real_t threshold{0.2};
    };
    class SigmoidActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return 1.0 / (1.0 + std::exp(-sigma * x));
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return sigma * sig * (1.0 - sig);
            }
            real_t sigma{1.0};
    };
    class BipolarSigmoidActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return 2.0 / (1.0 + std::exp(-sigma * x)) - 1.0;
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return 0.5 * sigma * (1.0 + sig) * (1.0 - sig);
            }
            real_t sigma{1.0};
    };
    class ReLUActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::max(static_cast<real_t>(0.0), x);
            }
            constexpr real_t derivative(const real_t x) const final {
                return x > 0.0 ? 1.0 : 0.0;
            }
    };
    class LeakyReLUActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= 0.0 ? x : 0.01 * x;
            }
            constexpr real_t derivative(const real_t x) const final {
                return x >= 0.0 ? 1.0 : 0.01;
            }
    };
    class ELUActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x >= 0.0 ? x : 0.01 * (std::exp(x) - 1.0);
            }
            constexpr real_t derivative(const real_t x) const final {
                return x >= 0.0 ? 1.0 : 0.01 * std::exp(x);
            }
    };
    class GaussianActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::exp(-std::pow(x - mean, 2) / (2 * sigma * sigma));
            }
            constexpr real_t derivative(const real_t x) const final {
                return - (x - mean) / (sigma * sigma) * (*this)(x);
            }
            real_t mean{0.0};
            real_t sigma{1.0};
    };
    class TanhActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::tanh(x);
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto t = (*this)(x);
                return 1.0 - t * t;
            }
    }; 
    class SoftplusActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return std::log(1.0 + std::exp(x));
            }
            constexpr real_t derivative(const real_t x) const final {
                return 1.0 / (1.0 + std::exp(-x));
            }
    };
    class SoftsignActivation : public Activation {
        public:
            constexpr real_t operator()(const real_t x) const final {
                return x / (1.0 + std::abs(x));
            }
            constexpr real_t derivative(const real_t x) const final {
                const auto denom = 1.0 + std::abs(x);
                return 1.0 / (denom * denom);
            }
    };
}
