#pragma once
#include "hnnet/activation.h"

namespace hNNet::Builtin {
    class IdentityActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x;
            }
            real_t derivative(const real_t x) const final {
                return 1.0;
            }
    };
    class LinearActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return slope * x + offset;
            }
            real_t derivative(const real_t x) const final {
                return slope;
            }
            real_t slope{1.0};
            real_t offset{0.0};
    };
    class StepActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x >= threshold ? 1.0 : 0.0;
            }
            real_t derivative(const real_t x) const final {
                return 0.0;
            }
            real_t threshold{0.0};
    };
    class BipolarStepActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x >= threshold ? +1.0 : -1.0;
            }
            real_t derivative(const real_t x) const final {
                return 0.0;
            }
            real_t threshold{0.0};
    };
    class PerceptronActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x > threshold ? +1 : x < -threshold ? -1 : 0;
            }
            real_t derivative(const real_t x) const final {
                return 0.0;
            }
            real_t threshold{0.2};
    };
    class SigmoidActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return 1.0 / (1.0 + std::exp(-sigma * x));
            }
            real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return sigma * sig * (1.0 - sig);
            }
            real_t sigma{1.0};
    };
    class BipolarSigmoidActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return 2.0 / (1.0 + std::exp(-sigma * x)) - 1.0;
            }
            real_t derivative(const real_t x) const final {
                const auto sig = (*this)(x);
                return 0.5 * sigma * (1.0 + sig) * (1.0 - sig);
            }
            real_t sigma{1.0};
    };
    class ReLUActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return std::max(static_cast<real_t>(0.0), x - offset);
            }
            real_t derivative(const real_t x) const final {
                return x > offset ? 1.0 : 0.0;
            }
            real_t offset{0.0};
    };
    class GaussianActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return std::exp(-std::pow(x - mean, 2) / (2 * sigma * sigma));
            }
            real_t derivative(const real_t x) const final {
                return - (x - mean) / (sigma * sigma) * (*this)(x);
            }
            real_t mean{0.0};
            real_t sigma{1.0};
    };
    class TanhActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return std::tanh(x);
            }
            real_t derivative(const real_t x) const final {
                const auto t = (*this)(x);
                return 1.0 - t * t;
            }
    };
    class LeakyReLUActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x >= 0.0 ? x : 0.01 * x;
            }
            real_t derivative(const real_t x) const final {
                return x >= 0.0 ? 1.0 : 0.01;
            }
    };
    class ELUActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x >= 0.0 ? x : 0.01 * (std::exp(x) - 1.0);
            }
            real_t derivative(const real_t x) const final {
                return x >= 0.0 ? 1.0 : 0.01 * std::exp(x);
            }
    };
    class SoftplusActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return std::log(1.0 + std::exp(x));
            }
            real_t derivative(const real_t x) const final {
                return 1.0 / (1.0 + std::exp(-x));
            }
    };
    class SoftsignActivation : public Activation {
        public:
            real_t operator()(const real_t x) const final {
                return x / (1.0 + std::abs(x));
            }
            real_t derivative(const real_t x) const final {
                const auto denom = 1.0 + std::abs(x);
                return 1.0 / (denom * denom);
            }
    };
}
