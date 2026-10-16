#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "../runtime.h"
#include "hnnet/types.h"

namespace hnnet::backprop::metal::optimizer {
    class Rprop {
        public:
            explicit Rprop(const real_t eta_plus, const real_t eta_minus)
                : _eta_plus(static_cast<float>(eta_plus)), _eta_minus(static_cast<float>(eta_minus)) {
                if (not std::isfinite(_eta_plus) or _eta_plus <= 1.0f) {
                    throw std::invalid_argument("Metal Rprop: eta_plus must be finite and > 1");
                }
                if (not std::isfinite(_eta_minus) or _eta_minus <= 0.0f or _eta_minus >= 1.0f) {
                    throw std::invalid_argument("Metal Rprop: eta_minus must be finite and between 0 and 1");
                }
            }

            void init(const detail::Context &context, const std::uint32_t weight_count) {
                _delta = context.buffer(std::size_t{weight_count} * sizeof(float));
                _previous_gradient = context.buffer(std::size_t{weight_count} * sizeof(float));
                std::fill_n(static_cast<float *>([_delta contents]), weight_count, 0.1f);
                _pipeline = context.pipeline("hnnet_rprop");
            }

            void update(id<MTLCommandBuffer> command, id<MTLBuffer> weights,
                        id<MTLBuffer> gradient, const std::uint32_t weight_count) {
                const detail::UpdateParameters parameters{weight_count, _eta_plus, _eta_minus};
                auto encoder = detail::encoder(command, _pipeline);
                [encoder setBuffer:weights offset:0 atIndex:0];
                [encoder setBuffer:gradient offset:0 atIndex:1];
                [encoder setBuffer:_delta offset:0 atIndex:2];
                [encoder setBuffer:_previous_gradient offset:0 atIndex:3];
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
                detail::dispatch(encoder, _pipeline, weight_count);
            }

        private:
            float _eta_plus;
            float _eta_minus;
            id<MTLBuffer> _delta = nil;
            id<MTLBuffer> _previous_gradient = nil;
            id<MTLComputePipelineState> _pipeline = nil;
    };
}