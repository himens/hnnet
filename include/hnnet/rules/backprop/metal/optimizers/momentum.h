#pragma once
#include <cmath>
#include <stdexcept>
#include "../runtime.h"
#include "hnnet/types.h"

namespace hnnet::backprop::metal::optimizer {
    class Momentum {
        public:
            explicit Momentum(const real_t learning_rate, const real_t momentum = 0.0_real)
                : _learning_rate(static_cast<float>(learning_rate)), _momentum(static_cast<float>(momentum)) {
                if (not std::isfinite(_learning_rate) or _learning_rate < 0.0f) {
                    throw std::invalid_argument("Metal Momentum: learning_rate must be finite and >= 0");
                }
                if (not std::isfinite(_momentum) or _momentum < 0.0f) {
                    throw std::invalid_argument("Metal Momentum: momentum must be finite and >= 0");
                }
            }

            void init(const detail::Context &context, const std::uint32_t weight_count) {
                _previous_delta = context.buffer(std::size_t{weight_count} * sizeof(float));
                _pipeline = context.pipeline("hnnet_momentum");
            }

            void update(id<MTLCommandBuffer> command, id<MTLBuffer> weights,
                        id<MTLBuffer> gradient, const std::uint32_t weight_count) {
                const detail::UpdateParameters parameters{weight_count, _learning_rate, _momentum};
                auto encoder = detail::encoder(command, _pipeline);
                [encoder setBuffer:weights offset:0 atIndex:0];
                [encoder setBuffer:gradient offset:0 atIndex:1];
                [encoder setBuffer:_previous_delta offset:0 atIndex:2];
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
                detail::dispatch(encoder, _pipeline, weight_count);
            }

        private:
            float _learning_rate;
            float _momentum;
            id<MTLBuffer> _previous_delta = nil;
            id<MTLComputePipelineState> _pipeline = nil;
    };
}