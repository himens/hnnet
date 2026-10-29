#pragma once
#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <utility>
#include "backend_type.h"
#include "hnnet/nnet.h"
#include "utils/math.h"

namespace hnnet::rule {
    ///////////////////
    // Backprop class //
    ////////////////////
    template <typename Backend, LossType Loss>
        class Backprop {
            public:
                // Constructor
                explicit Backprop(Backend backend, Loss loss, const int_t batch_size) : _backend(std::move(backend)), _loss(std::move(loss)), _batch_size(batch_size) {
                    if (batch_size == 0) {
                        throw std::invalid_argument("Backprop::Backprop: batch_size must not be zero");
                    }
                    std::println("Backprop::Backprop:: batch size: {}", batch_size);
                }
                // Initialize the rule
                template <std::derived_from<NNet::View> View>
                    void init(View view) {
                        _backend.init(view);
                    }
                // Learn from a whole epoch of training samples
                template <std::derived_from<NNet::View> View, DatasetType Inputs, DatasetType Targets>
                    requires backprop::BackendType<Backend, View, Inputs, Targets, Loss>
                    bool learn(View view, const Inputs &inputs, const Targets &targets) {
                        const auto sample_count = static_cast<index_t>(std::ranges::size(inputs));
                        if (sample_count == 0) {
                            throw std::invalid_argument("Backprop::learn: inputs must not be empty");
                        }
                        constexpr real_t loss_threshold{5e-3_real};
                        const auto batch_size = _batch_size < 0 ? sample_count : std::min(_batch_size, sample_count);
                        const auto batch_count = utils::math::ceil_div(sample_count, batch_size);
                        const backprop::TrainingData<Inputs, Targets> data{inputs, targets};
                        _backend.to_device(view, data);
                        real_t loss_value{0.0_real};
                        for (auto batch_index{0}; batch_index < batch_count; ++batch_index) {
                            const auto batch_begin = batch_index * batch_size;
                            const auto batch_end = std::min(sample_count, batch_begin + batch_size);
                            loss_value += _backend.backprop(view, batch_begin, batch_end, _loss);
                        }
                        _backend.to_host(view, view.weights());
                        std::println("Backprop::learn: average loss: {:.5f}", loss_value / sample_count);
                        return loss_value / sample_count < loss_threshold;
                    }
            private:
                // Data members
                Backend _backend;
                Loss _loss;
                int_t _batch_size;
        };
}
