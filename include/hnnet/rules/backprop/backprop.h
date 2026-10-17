#pragma once
#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <utility>
#include "backend_type.h"
#include "utils/math.h"

namespace hnnet::rule {
    ///////////////////
    // Backprop class //
    ////////////////////
    template <typename Backend, LossType Loss>
        class Backprop {
            public:
                // Constructor
                explicit Backprop(Backend backend,
                                  Loss loss,
                                  const int_t batch_size) : _backend(std::move(backend)), _loss(std::move(loss)), _batch_size(batch_size) {
                    if (batch_size == 0) {
                        throw std::invalid_argument("Backprop::Backprop: batch_size must not be zero");
                    }
                    std::println("Backprop::Backprop:: batch size: {}", batch_size);
                }
                // Initialize the rule
                void init(auto &net) {
                    _backend.init(net);
                }
                // Learn from a whole epoch of training samples
                template <typename Net, DatasetType Inputs, DatasetType Targets>
                    requires backprop::BackendType<Backend, Net, Inputs, Targets, Loss>
                    bool learn(Net &net, const Inputs &inputs, const Targets &targets) {
                        const auto sample_count = static_cast<index_t>(std::ranges::size(inputs));
                        if (sample_count == 0) {
                            throw std::invalid_argument("Backprop::learn: inputs must not be empty");
                        }
                        constexpr real_t loss_threshold{5e-3_real};
                        const auto batch_size = _batch_size < 0 ? sample_count : std::min(_batch_size, sample_count);
                        const auto batch_count = utils::math::ceil_div(sample_count, batch_size);
                        _backend.to_device(net, inputs, targets);
                        real_t loss{0.0_real};
                        for (auto ibatch{0}; ibatch < batch_count; ++ibatch) {
                            const auto batch_begin = ibatch * batch_size;
                            const auto batch_end = std::min(sample_count, batch_begin + batch_size);
                            loss += _backend.backprop(net, batch_begin, batch_end, _loss);
                        }
                        _backend.to_host(net);
                        std::println("Backprop::learn: average loss: {:.5f}", loss / sample_count);
                        return loss / sample_count < loss_threshold;
                    }
            private:
                // Data members
                Backend _backend;
                Loss _loss;
                int_t _batch_size;
        };
}
