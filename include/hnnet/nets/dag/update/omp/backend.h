#pragma once
#include <iterator>
#include <span>
#include "hnnet/nets/dag/dag_net.h"

namespace hnnet::dag::update::omp {
    ///////////////////
    // Backend class //
    ///////////////////
    class Backend {
        public:
            // Initialize the backend
            void init(net::DAGNet::View &view) {
                _view = view;
            }
            // Transfer the network to the device
            void to_device(net::DAGNet::View &view, std::span<NNet::State> states) {
                _view = view;
                _states = states;
            }
            // Transfer the network back to the host
            void to_host(net::DAGNet::View &, std::span<NNet::State>) {}
            // Update the states by propagating signals through the net
            void update() const {
                const auto state_count = std::ssize(_states);
                if (state_count == 1) {
                    update_state(_states.front());
                }
                else if (state_count > 1) {
                    #pragma omp parallel for
                    for (index_t state_index = 0; state_index < state_count; ++state_index) {
                        update_state(_states[state_index]);
                    }
                }
            }
        private:
            // Propagate signals through one state
            void update_state(NNet::State &state) const {
                const auto partitions = _view.partitions();
                const auto partition_layer_indices = _view.partition_layer_indices();
                const auto layers = _view.layers();
                for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    const auto &partition = partitions[partition_index];
                    const auto layer_index = partition_layer_indices[partition_index];
                    if (layer_index != net::DAGNet::Layer::no_layer) {
                        const auto &layer = layers[layer_index];
                        update_layer(state, layer);
                        partition_index += layer.rx_count - 1;  // layer members are contiguous: skip them all at once
                    }
                    else {
                        update_partition(state, partition);
                    }
                }
            }
            // Propagate signals through one sparse partition
            void update_partition(NNet::State &state, const NNet::Partition &partition) const {
                real_t weighted_sum{0.0_real};
                const auto weights = _view.weights();
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
            void update_layer(NNet::State &state, const net::DAGNet::Layer &layer) const {
                const auto weights = _view.weights();
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
            std::span<NNet::State> _states{};
    };
}
