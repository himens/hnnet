#pragma once
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>
#include "hnnet/types.h"

namespace hnnet::backprop::omp::optimizer {
    ////////////////////
    // Momentum class //
    ////////////////////
    class Momentum {
        public:
            // Constructor
            explicit Momentum(const real_t learning_rate, const real_t momentum = 0.0_real) : _learning_rate(learning_rate), _momentum(momentum) {
                if (learning_rate < 0.0_real) {
                    throw std::invalid_argument("Momentum: learning_rate must be >= 0");
                }
                if (momentum < 0.0_real) {
                    throw std::invalid_argument("Momentum: momentum must be >= 0");
                }
                std::println("Momentum::Momentum: learning_rate: {}, momentum: {}", _learning_rate, _momentum);
            }
            // Initialize the state
            void init(const int_t weight_count) {
                _previous_delta.assign(weight_count, 0.0_real);
            }
            // Update the weights
            void update(std::span<real_t> weights, std::span<const real_t> gradient) {
                //#pragma omp parallel for
                for (index_t connection_index = 0; connection_index < std::ssize(gradient); ++connection_index) {
                    const auto delta = - (_learning_rate * gradient[connection_index]) + (_momentum * _previous_delta[connection_index]);
                    weights[connection_index] += delta;
                    _previous_delta[connection_index] = delta;
                }
            }
        private:
            // Data members
            real_t _learning_rate;
            real_t _momentum;
            std::vector<real_t> _previous_delta;
    };
}
