#pragma once
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>
#include "hnnet/types.h"
#include "utils/math.h"

namespace hnnet::builtin::backend::omp::backprop::optimizer {
    /////////////////
    // Rprop class //
    /////////////////
    // NOTE: Rprop comes in several variants. This class implements iRprop-
    class Rprop {
        public:
            // Constructor
            explicit Rprop(const real_t eta_plus, const real_t eta_minus) : _eta_plus(eta_plus), _eta_minus(eta_minus) {
                if (eta_plus <= 1.0_real) {
                    throw std::invalid_argument("Rprop: eta_plus must be > 1");
                }
                if (eta_minus <= 0.0_real or eta_minus >= 1.0_real) {
                    throw std::invalid_argument("Rprop: eta_minus must be between 0 and 1");
                }
                std::println("Rprop::Rprop: eta_plus: {}, eta_minus: {}", _eta_plus, _eta_minus);
            }
            // Initialize the state
            void init(const int_t weight_count) {
                _delta.assign(weight_count, 0.1_real);
                _previous_gradient.assign(weight_count, 0.0_real);
            }
            // Update the weights
            void update(std::span<real_t> weights, std::span<const real_t> gradient) {
                #pragma omp parallel for
                for (index_t iconn = 0; iconn < std::ssize(gradient); ++iconn) {
                    constexpr real_t delta_min{1e-6_real};
                    constexpr real_t delta_max{50.0_real};
                    auto &delta = _delta[iconn];
                    auto &previous_gradient = _previous_gradient[iconn];
                    const auto &grad = gradient[iconn];
                    const auto sign_product = previous_gradient * grad;
                    if (sign_product > 0_real) {
                        delta = utils::math::min(_eta_plus * delta, delta_max);
                        weights[iconn] += -utils::math::sign(grad) * delta;
                        previous_gradient = grad;
                    }
                    else if (sign_product < 0_real) {
                        delta = utils::math::max(_eta_minus * delta, delta_min);
                        previous_gradient = 0.0_real;
                    }
                    else {
                        weights[iconn] += -utils::math::sign(grad) * delta;
                        previous_gradient = grad;
                    }
                }
            }
        private:
            // Data members
            real_t _eta_plus;
            real_t _eta_minus;
            std::vector<real_t> _delta;
            std::vector<real_t> _previous_gradient;
    };
}
