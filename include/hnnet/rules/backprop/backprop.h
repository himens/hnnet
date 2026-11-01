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
                template <std::derived_from<NNet::View> View, TrainingDataType Data>
                    requires backprop::BackendType<Backend, View, Data, Loss>
                    void init(View view, const Data &data) {
                        _input_count = std::ssize(data.inputs);
                        if (_input_count == 0) {
                            throw std::invalid_argument("Backprop::init: inputs must not be empty");
                        }
                        if (std::ranges::size(data.inputs) != std::ranges::size(data.targets)) {
                            throw std::invalid_argument("Backprop::init: input and target sample counts must match");
                        }
                        for (const auto &[input, target] : std::views::zip(data.inputs, data.targets)) {
                            if (std::ssize(input) != view.input_neuron_count() or
                                std::ssize(target) != view.output_neuron_count()) {
                                throw std::invalid_argument("Backprop::init: input or target size does not match the network");
                            }
                        }
                        _backend.init(view);
                        _backend.to_device(data);
                    }
                // Learn from a whole epoch of training samples
                bool learn() {
                    constexpr real_t loss_threshold{5e-3_real};
                    const auto batch_size = _batch_size < 0 ? _input_count : std::min(_batch_size, _input_count);
                    const auto batch_count = utils::math::ceil_div(_input_count, batch_size);
                    real_t loss_value{0.0_real};
                    for (auto batch_index{0}; batch_index < batch_count; ++batch_index) {
                        const auto batch_begin = batch_index * batch_size;
                        const auto batch_end = std::min(_input_count, batch_begin + batch_size);
                        loss_value += _backend.backprop(batch_begin, batch_end, _loss);
                    }
                    std::println("Backprop::learn: average loss: {:.5f}", loss_value / _input_count);
                    return loss_value / _input_count < loss_threshold;
                }
                // Apply the learned weights
                void apply(std::vector<real_t> &weights) {
                    _backend.to_host(weights);
                }
            private:
                // Data members
                Backend _backend;
                Loss _loss;
                int_t _batch_size;
                index_t _input_count{0};
        };
}
