#pragma once
#include <algorithm>
#include <array>
#include <iterator>
#include <span>
#include <utility>
#include <omp.h>
#include "hnnet/nets/dag/dag_net.h"

namespace hnnet::dag::update::omp {
    ///////////////////
    // Backend class //
    ///////////////////
    class Backend {
        public:
            // Initialize the backend
            void init(net::DAGNet::View view) {
                _view = view;
                _states.clear();
                _inputs.clear();
                _outputs.clear();
            }
            // Transfer data to the device
            void to_device(const DatasetType auto &inputs) {
                for (const auto &input : inputs) {
                    if (std::ssize(input) != _view.input_neuron_count()) {
                        throw std::invalid_argument("Backend::to_device: input or state size does not match the network");
                    }
                }
                _inputs.clear();
                _inputs.reserve(std::ranges::size(inputs));
                for (const auto &input : inputs) {
                    _inputs.emplace_back(input);
                }
                _states.clear();
                _outputs.clear();
            }
            // Transfer data to the host
            void to_host(std::span<const NNet::State> &states) const {
                states = _states;
            }
            void to_host(dataset_vector_t &outputs) {
                outputs = std::move(_outputs);
            }
            // Seed and update a batch; result index i corresponds to input batch_begin + i
            void update_states(const DataType auto &weights, index_t batch_begin = 0, index_t batch_end = -1) {
                const auto input_count = std::ssize(_inputs);
                if (batch_end == -1) {
                    batch_end = input_count;
                }
                if (batch_begin < 0 or batch_end < batch_begin or batch_end > input_count) {
                    throw std::invalid_argument("Backend::update_states: batch range out of bounds");
                }
                if (std::ssize(weights) != _view.connection_count()) {
                    throw std::invalid_argument("Backend::update_states: weight counts must match");
                }
                const auto batch_size = batch_end - batch_begin;
                if (std::ssize(_states) > batch_size) {
                    _states.erase(_states.begin() + batch_size, _states.end());
                }
                _states.reserve(batch_size);
                while (std::ssize(_states) < batch_size) {
                    _states.emplace_back(_view.neuron_count());
                }
                _outputs.resize(batch_size);
                for (auto &output : _outputs) {
                    output.resize(_view.output_neuron_count());
                }
                #pragma omp parallel for if(batch_size > 1)
                for (index_t index = 0; index < batch_size; ++index) {
                    auto &state = _states[index];
                    state.seed(_view, _inputs[batch_begin + index]);
                    update_state(state, _outputs[index], weights);
                }
            }
        private:
            // Propagate signals through one state
            void update_state(NNet::State &state, std::vector<real_t> &output, std::span<const real_t> weights) const {
                const auto partitions = _view.partitions();
                const auto partition_layer_indices = _view.partition_layer_indices();
                const auto layers = _view.layers();
                const auto signals = state.signals();
                for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    const auto &partition = partitions[partition_index];
                    const auto layer_index = partition_layer_indices[partition_index];
                    if (layer_index != net::DAGNet::Layer::no_layer) {
                        const auto &layer = layers[layer_index];
                        update_layer(state, layer, weights);
                        partition_index += layer.rx_count - 1;  // layer members are contiguous: skip them all at once
                    }
                    else {
                        update_partition(state, partition, weights);
                    }
                }
                const auto output_neurons_indices = _view.output_neurons_indices();
                index_t output_index{0};
                for (; output_index <= (std::ssize(output_neurons_indices) - simd_register_size); output_index += simd_register_size) {
                    output[output_index]     = signals[output_neurons_indices[output_index]];
                    output[output_index + 1] = signals[output_neurons_indices[output_index + 1]];
                    output[output_index + 2] = signals[output_neurons_indices[output_index + 2]];
                    output[output_index + 3] = signals[output_neurons_indices[output_index + 3]];
                }
                for (; output_index < std::ssize(output_neurons_indices); ++output_index) {
                    output[output_index] = signals[output_neurons_indices[output_index]];
                }
            }
            // Propagate signals through one sparse partition
            void update_partition(NNet::State &state, const NNet::Partition &partition, std::span<const real_t> weights) const {
                real_t weighted_sum{0.0_real};
                const auto connections = _view.connections();
                const auto neurons = _view.neurons();
                auto signals = state.signals();
                auto weighted_sums = state.weighted_sums();
                auto connection_index = partition.connection_begin;
                for (; connection_index <= (partition.connection_end - simd_register_size); connection_index += simd_register_size) {
                    weighted_sum +=
                        + weights[connection_index]     * signals[connections[connection_index].tx_index]
                        + weights[connection_index + 1] * signals[connections[connection_index + 1].tx_index]
                        + weights[connection_index + 2] * signals[connections[connection_index + 2].tx_index]
                        + weights[connection_index + 3] * signals[connections[connection_index + 3].tx_index];
                }
                //#pragma omp simd reduction(+:weighted_sum)
                for (; connection_index < partition.connection_end; ++connection_index) {
                    weighted_sum += weights[connection_index] * signals[connections[connection_index].tx_index];
                }
                weighted_sums[partition.rx_index] = weighted_sum;
                signals[partition.rx_index] = neurons[partition.rx_index].activate(weighted_sum);
            }
            // Propagate signals through one dense layer
            void update_layer(NNet::State &state, const net::DAGNet::Layer &layer, std::span<const real_t> weights) const {
                const auto neurons = _view.neurons();
                auto signals = state.signals();
                auto weighted_sums = state.weighted_sums();
                for (auto row_index{0}; row_index < layer.rx_count; ++row_index) {
                    const auto row_offset = layer.weight_offset + row_index * layer.tx_count;
                    real_t weighted_sum{0.0_real};
                    index_t column_index{0};
                    for (; column_index <= (layer.tx_count - simd_register_size); column_index += simd_register_size) {
                        weighted_sum +=
                            + weights[row_offset + column_index]     * signals[layer.tx_begin + column_index]
                            + weights[row_offset + column_index + 1] * signals[layer.tx_begin + column_index + 1]
                            + weights[row_offset + column_index + 2] * signals[layer.tx_begin + column_index + 2]
                            + weights[row_offset + column_index + 3] * signals[layer.tx_begin + column_index + 3];
                    }
                    //#pragma omp simd reduction(+:weighted_sum)
                    for (; column_index < layer.tx_count; ++column_index) {
                        weighted_sum += weights[row_offset + column_index] * signals[layer.tx_begin + column_index];
                    }
                    weighted_sums[layer.rx_begin + row_index] = weighted_sum;
                    signals[layer.rx_begin + row_index] = neurons[layer.rx_begin + row_index].activate(weighted_sum);
                }
            }
        private:
            // Data members
            net::DAGNet::View _view{};
            std::vector<std::span<const real_t>> _inputs{};
            std::vector<NNet::State> _states{};
            dataset_vector_t _outputs{};
    };
}
