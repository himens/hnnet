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
                _states.assign(omp_get_max_threads(), NNet::State{view.neuron_count()});
                _inputs.clear();
                _outputs.clear();
            }
            // Transfer data to the device
            void to_device(const DatasetType auto &samples) {
                for (const auto &inputs : samples) {
                    if (std::ssize(inputs) != _view.input_neuron_count()) {
                        throw std::invalid_argument("Backend::to_device: input or state size does not match the network");
                    }
                }
                _inputs.clear();
                _inputs.reserve(std::ranges::size(samples));
                for (const auto &inputs : samples) {
                    _inputs.emplace_back(inputs);
                }
                _outputs.resize(_inputs.size());
                for (auto &output : _outputs) {
                    output.resize(_view.output_neuron_count());
                }
            }
            // Transfer data to the host
            void to_host(std::span<const NNet::State> &states) const {
                states = _states;
            }
            void to_host(std::vector<data_vector_t> &outputs) {
                outputs = std::move(_outputs);
            }
            // Seed and update all samples or one sample using supplied weights
            void update_states(const DataType auto &weights, index_t sample_index = -1) {
                const auto sample_count = std::ssize(_inputs);
                if (sample_index < -1 or sample_index >= sample_count) {
                    throw std::invalid_argument("Backend::update_states: sample index out of range");
                }
                if (std::ssize(weights) != _view.connection_count()) {
                    throw std::invalid_argument("Backend::update_states: weight counts must match");
                }
                if (sample_index == -1) {
                    #pragma omp parallel for if(sample_count > 1) num_threads(std::ssize(_states))
                    for (index_t index = 0; index < sample_count; ++index) {
                        auto &state = _states[omp_get_thread_num()];
                        _view.seed_state(state, _inputs[index]);
                        update_state(state, _outputs[index], weights);
                    }
                }
                else {
                    auto &state = _states[omp_get_thread_num()];
                    _view.seed_state(state, _inputs[sample_index]);
                    update_state(state, _outputs[sample_index], weights);
                }
            }
        private:
            // Propagate signals through one state
            void update_state(NNet::State &state, data_vector_t &output, std::span<const real_t> weights) {
                const auto partitions = _view.partitions();
                const auto partition_layer_indices = _view.partition_layer_indices();
                const auto layers = _view.layers();
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
                for (const auto &[output_index, neuron_index] : _view.output_neurons_indices() | std::views::enumerate) {
                    output[output_index] = state.signals[neuron_index]; // TODO: it can be optimized using SIMD or parallel reduction?
                }
            }
            // Propagate signals through one sparse partition
            void update_partition(NNet::State &state, const NNet::Partition &partition, std::span<const real_t> weights) {
                real_t weighted_sum{0.0_real};
                const auto connections = _view.connections();
                const auto neurons = _view.neurons();
                auto connection_index = partition.connection_begin;
                for (; connection_index <= (partition.connection_end - simd_register_size); connection_index += simd_register_size) {
                    weighted_sum +=
                        + weights[connection_index]     * state.signals[connections[connection_index].tx_index]
                        + weights[connection_index + 1] * state.signals[connections[connection_index + 1].tx_index]
                        + weights[connection_index + 2] * state.signals[connections[connection_index + 2].tx_index]
                        + weights[connection_index + 3] * state.signals[connections[connection_index + 3].tx_index];
                }
                //#pragma omp simd reduction(+:weighted_sum)
                for (; connection_index < partition.connection_end; ++connection_index) {
                    weighted_sum += weights[connection_index] * state.signals[connections[connection_index].tx_index];
                }
                state.weighted_sums[partition.rx_index] = weighted_sum;
                state.signals[partition.rx_index] = neurons[partition.rx_index].activate(weighted_sum);
            }
            // Propagate signals through one dense layer
            void update_layer(NNet::State &state, const net::DAGNet::Layer &layer, std::span<const real_t> weights) const {
                const auto neurons = _view.neurons();
                for (auto row_index{0}; row_index < layer.rx_count; ++row_index) {
                    real_t weighted_sum{0.0_real};
                    const auto row_offset = layer.weight_offset + row_index * layer.tx_count;
                    index_t column_index{0};
                    for (; column_index <= (layer.tx_count - simd_register_size); column_index += simd_register_size) {
                        weighted_sum +=
                            + weights[row_offset + column_index]     * state.signals[layer.tx_begin + column_index]
                            + weights[row_offset + column_index + 1] * state.signals[layer.tx_begin + column_index + 1]
                            + weights[row_offset + column_index + 2] * state.signals[layer.tx_begin + column_index + 2]
                            + weights[row_offset + column_index + 3] * state.signals[layer.tx_begin + column_index + 3];
                    }
                    //#pragma omp simd reduction(+:weighted_sum)
                    for (; column_index < layer.tx_count; ++column_index) {
                        weighted_sum += weights[row_offset + column_index] * state.signals[layer.tx_begin + column_index];
                    }
                    state.weighted_sums[layer.rx_begin + row_index] = weighted_sum;
                    state.signals[layer.rx_begin + row_index] = neurons[layer.rx_begin + row_index].activate(weighted_sum);
                }
            }
        private:
            // Data members
            net::DAGNet::View _view{};
            std::vector<std::span<const real_t>> _inputs{};
            std::vector<NNet::State> _states{};
            std::vector<data_vector_t> _outputs{};
    };
}
