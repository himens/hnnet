#pragma once
#include "hnnet/types.h"

namespace hNNet {
    template <typename T>
        concept ActivationType = requires(T activation, const real_t x) {
            { activation(x) } -> std::same_as<real_t>;
            { activation.derivative(x) } -> std::same_as<real_t>;
        };
    struct IdentityActivation {
        real_t operator()(const real_t x) const {
            return x;
        }
        real_t derivative(const real_t x) const {
            return 1.0;
        }
    };
    struct LinearActivation {
        real_t operator()(const real_t x) const {
            return slope * x + offset;
        }
        real_t derivative(const real_t x) const {
            return slope;
        }
        real_t slope{1.0};
        real_t offset{0.0};
    };
    struct StepActivation {
        real_t operator()(const real_t x) const {
            return x >= threshold ? 1.0 : 0.0;
        }
        real_t derivative(const real_t x) const {
            return 0.0;
        }
        real_t threshold{0.0};
    };
    struct BipolarStepActivation {
        real_t operator()(const real_t x) const {
            return x >= threshold ? +1.0 : -1.0;
        }
        real_t derivative(const real_t x) const {
            return 0.0;
        }
        real_t threshold{0.0};
    };
    struct PerceptronActivation {
        real_t operator()(const real_t x) const {
            return x > threshold ? +1 : x < -threshold ? -1 : 0;
        }
        real_t derivative(const real_t x) const {
            return 0.0;
        }
        real_t threshold{0.2};
    };
    struct SigmoidActivation {
        real_t operator()(const real_t x) const {
            return 1.0 / (1.0 + std::exp(-sigma * x));
        }
        real_t derivative(const real_t x) const {
            const auto sig = (*this)(x);
            return sigma * sig * (1.0 - sig);
        }
        real_t sigma{1.0};
    };
    struct BipolarSigmoidActivation {
        real_t operator()(const real_t x) const {
            return 2.0 / (1.0 + std::exp(-sigma * x)) - 1.0;
        }
        real_t derivative(const real_t x) const {
            const auto sig = (*this)(x);
            return 0.5 * sigma * (1.0 + sig) * (1.0 - sig);
        }
        real_t sigma{1.0};
    };
    struct ReLUActivation {
        real_t operator()(const real_t x) const {
            return std::max(offset, x);
        }
        real_t derivative(const real_t x) const {
            return x > offset ? 1.0 : 0.0;
        }
        real_t offset{0.0};
    };
    struct GaussianActivation {
        real_t operator()(const real_t x) const {
            return std::exp(-std::pow(x - mean, 2) / (2 * sigma * sigma));
        }
        real_t derivative(const real_t x) const {
            return - (x - mean) / (sigma * sigma) * (*this)(x);
        }
        real_t mean{0.0};
        real_t sigma{1.0};
    };
    struct TanhActivation {
        real_t operator()(const real_t x) const {
            return std::tanh(x);
        }
        real_t derivative(const real_t x) const {
            const auto t = (*this)(x);
            return 1.0 - t * t;
        }
    };
    struct LeakyReLUActivation {
        real_t operator()(const real_t x) const {
            return x >= 0.0 ? x : 0.01 * x;
        }
        real_t derivative(const real_t x) const {
            return x >= 0.0 ? 1.0 : 0.01;
        }
    };
    struct ELUActivation {
        real_t operator()(const real_t x) const {
            return x >= 0.0 ? x : 0.01 * (std::exp(x) - 1.0);
        }
        real_t derivative(const real_t x) const {
            return x >= 0.0 ? 1.0 : 0.01 * std::exp(x);
        }
    };
    struct SwishActivation {
        real_t operator()(const real_t x) const {
            return x * SigmoidActivation()(x);
        }
        real_t derivative(const real_t x) const {
            const auto sig = SigmoidActivation()(x);
            return sig + x * sig * (1.0 - sig);
        }
    };
    struct SoftplusActivation {
        real_t operator()(const real_t x) const {
            return std::log(1.0 + std::exp(x));
        }
        real_t derivative(const real_t x) const {
            return 1.0 / (1.0 + std::exp(-x));
        }
    };
    struct SoftsignActivation {
        real_t operator()(const real_t x) const {
            return x / (1.0 + std::abs(x));
        }
        real_t derivative(const real_t x) const {
            const auto denom = 1.0 + std::abs(x);
            return 1.0 / (denom * denom);
        }
    };
    using Activation = std::variant<IdentityActivation,
                                    LinearActivation,
                                    StepActivation,
                                    BipolarStepActivation,
                                    PerceptronActivation,
                                    SigmoidActivation,
                                    BipolarSigmoidActivation,
                                    ReLUActivation,
                                    GaussianActivation,
                                    TanhActivation,
                                    LeakyReLUActivation,
                                    ELUActivation,
                                    SwishActivation,
                                    SoftplusActivation,
                                    SoftsignActivation>;
}
