#pragma once
#include "hnnet/builtin/backprop/optimizer.h"

namespace hnnet::builtin {
    ////////////////////
    // Momentum class //
    ////////////////////
    class Momentum {
        public:
            struct State {
                State() = default;
                explicit State(const int_t weight_count) : prev_dweights(weight_count, 0.0_real) {}
                grad_vector_t prev_dweights;
            };
            // Constructor
            explicit Momentum(const real_t learning_rate, const real_t momentum = 0.0_real) : _learning_rate(learning_rate), _momentum(momentum) {
                if (_learning_rate < 0.0_real) {
                    throw std::invalid_argument("Momentum::Momentum: learning_rate must be >= 0");
                }
                if (_momentum < 0.0_real) {
                    throw std::invalid_argument("Momentum::Momentum: momentum must be >= 0");
                }
                std::println("Momentum::Momentum: learning_rate: {}, momentum: {}", _learning_rate, _momentum);
            }
            // Initialize the state
            void init(const int_t weight_count) {
                _state = State{weight_count};
            }
            // Update the weights
            void update(DAGNet::View &view, const grad_vector_t &gradient) {
                for (auto iconn{0}; iconn < std::ssize(gradient); ++iconn) {
                    const auto dweight = - (_learning_rate * gradient[iconn]) + (_momentum * _state.prev_dweights[iconn]);
                    view.weight(iconn) += dweight;
                    _state.prev_dweights[iconn] = dweight;
                }
            }
        private:
            // Data members
            real_t _learning_rate{0.0_real};
            real_t _momentum{0.0_real};
            State _state{};
    };
    /////////////////
    // Rprop class //
    /////////////////
    // NOTE: Rprop comes in several variants. This class implements iRprop-
    class Rprop {
        public:
            struct State {
                State() = default;
                explicit State(const int_t weight_count) : deltas(weight_count, 0.1_real), prev_gradient(weight_count, 0.0_real) {}
                std::vector<real_t> deltas;
                grad_vector_t prev_gradient;
            };
            // Constructor
            explicit Rprop(const real_t eta_plus, const real_t eta_minus) : _eta_plus(eta_plus), _eta_minus(eta_minus) {
                if (_eta_plus <= 1.0_real) {
                    throw std::invalid_argument("Rprop::Rprop: eta_plus must be > 1");
                }
                if (_eta_minus <= 0.0_real or _eta_minus >= 1.0_real) {
                    throw std::invalid_argument("Rprop::Rprop: eta_minus must between 0 and 1");
                }
                std::println("Rprop::Rprop: eta_plus: {}, eta_minus: {}", _eta_plus, _eta_minus);
            }
            // Initialize the state
            void init(const int_t weight_count) {
                _state = State{weight_count};
            }
            // Update the weights
            void update(DAGNet::View &view, const grad_vector_t &gradient) {
                for (auto iconn{0}; iconn < std::ssize(gradient); ++iconn) {
                    constexpr real_t delta_min{1e-6_real};
                    constexpr real_t delta_max{50.0_real};
                    auto &delta = _state.deltas[iconn];
                    auto &prev_grad = _state.prev_gradient[iconn];
                    const auto &grad = gradient[iconn];
                    const auto sign_prod = prev_grad * grad;
                    if (sign_prod > 0_real) {
                        delta = utils::math::min(_eta_plus * delta, delta_max);
                        const auto dweight = - utils::math::sign(grad) * delta;
                        view.weight(iconn) += dweight;
                        prev_grad = grad;
                    }
                    else if (sign_prod < 0_real) {
                        delta = utils::math::max(_eta_minus * delta, delta_min);
                        prev_grad = 0.0_real;
                    }
                    else {
                        const auto dweight = - utils::math::sign(grad) * delta;
                        view.weight(iconn) += dweight;
                        prev_grad = grad;
                    }
                }
            }
        private:
            // Data members
            real_t _eta_plus{0.0_real};
            real_t _eta_minus{0.0_real};
            State _state{};
    };
}
