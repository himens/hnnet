#pragma once
#include <algorithm>
#include <iterator>
#include <numeric>
#include <print>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hnnet/learning_rule_type.h"
#include "hnnet/nnet.h"

namespace hnnet::net {
    //////////////////
    // DAGNet class //
    //////////////////
    class DAGNet : public NNet {
        public:
            // Data types
            struct Layer {
                static constexpr index_t no_layer{-1};
                index_t tx_begin;
                int_t tx_count;
                index_t rx_begin;
                int_t rx_count;
                index_t weight_offset;
            };
            class View : public NNet::View {
                public:
                    bool is_dense(index_t partition_index) const { // TODO: cambiare nome?
                        return _net._layer_indices[partition_index] != Layer::no_layer;
                    }
                    std::span<const Layer> layers() const {
                        return _net._layers;
                    }
                private:
                    friend class DAGNet;
                    explicit View(DAGNet &net) : NNet::View(net), _net(net) {}
                    DAGNet &_net;
            };
        public:
            // Constructor
            DAGNet(Topology topology) : NNet(std::move(topology)) {
                init();
            }
            // Destructor
            virtual ~DAGNet() = default;
            // Create a view of the net
            View view() {
                return View(*this);
            }
            // Train net using a set of training samples
            template <DatasetType Inputs, DatasetType Targets, LearningRuleType<DAGNet, Inputs, Targets> LearningRule>
                void train(const Inputs &inputs, const Targets &targets, LearningRule rule) {
                    NNet::train(*this, inputs, targets, rule);
                }
        protected:
            // Update net state
            void update_state(State &state) const override {
                propagate_signals(state);
            }
        private:
            // Data types
            class UnionFind {
                public:
                    explicit UnionFind(const int_t count) : _root_indices(count), _ranks(count, 0) {
                        if (count < 0) {
                            throw std::invalid_argument("UnionFind: invalid count!");
                        }
                        std::iota(std::begin(_root_indices), std::end(_root_indices), 0);
                    }
                    index_t find(const index_t index) {
                        if (_root_indices[index] == index) {
                            return index;
                        }
                        return _root_indices[index] = find(_root_indices[index]);
                    }
                    void unite(const index_t lhs_index, const index_t rhs_index) {
                        auto lhs_root_index = find(lhs_index);
                        auto rhs_root_index = find(rhs_index);
                        if (lhs_root_index == rhs_root_index) {
                            return;
                        }
                        if (_ranks[lhs_root_index] < _ranks[rhs_root_index]) {
                            std::swap(lhs_root_index, rhs_root_index);
                        }
                        _root_indices[rhs_root_index] = lhs_root_index;
                        if (_ranks[lhs_root_index] == _ranks[rhs_root_index]) {
                            _ranks[lhs_root_index]++;
                        }
                    }
                private:
                    index_vector_t _root_indices;
                    std::vector<int_t> _ranks;
            };
            // Initialize net
            void init() {
                const auto connections = this->connections();
                const auto neurons = this->neurons();
                auto partitions = this->partitions();
                // topologically order partitions (Kahn's algorithm)
                std::vector<index_vector_t> rx_indices(neurons.size());
                for (const auto &[tx_index, rx_index] : connections) {
                    rx_indices[tx_index].push_back(rx_index);
                }
                std::vector<int_t> visits_left(neurons.size(), 0);
                for (const auto &[rx_index, connection_begin, connection_end] : partitions) {
                    visits_left[rx_index] = connection_end - connection_begin;
                }
                index_vector_t visited_indices;
                for (auto neuron_index{0}; neuron_index < std::ssize(neurons); ++neuron_index) {
                    if (visits_left[neuron_index] == 0) {
                        visited_indices.push_back(neuron_index);
                    }
                }
                int_t visited_count{0};
                std::vector<int_t> topo_ranks(neurons.size(), 0);
                while (not visited_indices.empty()) {
                    const auto visited_index = visited_indices.back();
                    visited_indices.pop_back();
                    topo_ranks[visited_index] = visited_count++;
                    for (const auto &rx_index : rx_indices[visited_index]) {
                        if (--visits_left[rx_index] == 0) {
                            visited_indices.push_back(rx_index);
                        }
                    }
                }
                if (visited_count != std::ssize(neurons)) {
                    throw std::runtime_error("DAGNet::init: net contains a cycle, topological order does not exist!");
                }
                std::ranges::sort(partitions, [&] (const auto &lhs, const auto &rhs) { return topo_ranks[lhs.rx_index] < topo_ranks[rhs.rx_index]; });
                // group partitions sharing the exact same (contiguous) tx range
                _layer_indices.assign(std::ssize(partitions), Layer::no_layer);
                UnionFind union_find(partitions.size());
                std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                for (const auto &[partition_index, partition] : partitions | std::views::enumerate) {
                    const auto count = (partition.connection_end - partition.connection_begin);
                    const auto tx_begin = connections[partition.connection_begin].tx_index;
                    if ((connections[partition.connection_end - 1].tx_index - tx_begin) != count - 1) {
                        continue;  // tx range has gaps: not a pure dense candidate
                    }
                    const auto [it, inserted] = hash_map.try_emplace(std::pair{tx_begin, count}, partition_index);
                    if (not inserted) {
                        union_find.unite(it->second, partition_index);
                    }
                }
                std::vector<index_vector_t> groups(partitions.size());
                for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    const auto root_index = union_find.find(partition_index);
                    groups[root_index].push_back(partition_index);
                }
                // find dense layers
                _layers.clear();
                for (index_t partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    auto partition_indices = std::exchange(groups[union_find.find(partition_index)], index_vector_t{});
                    if (partition_indices.empty()) {
                        continue;
                    }
                    // TODO: Support single-receiver dense layers, excluding sparse singletons also present in groups.
                    if (partition_indices.size() < 2) {
                        continue;  // no gain grouping a single receiver
                    }
                    const auto [min_partition_index, max_partition_index] = std::ranges::minmax(partition_indices);
                    if ((max_partition_index - min_partition_index + 1) != std::ssize(partition_indices)) {
                        continue; // must be contiguous partitions (in topo order)
                    }
                    // sort partition indices per increasing receiver and check their ranges are contiguous
                    std::ranges::sort(partition_indices, [&] (const auto &lhs, const auto &rhs) { return partitions[lhs].rx_index < partitions[rhs].rx_index; });
                    auto contiguous = true;
                    for (auto member_index{1}; member_index < std::ssize(partition_indices); ++member_index) {
                        const auto &prev = partitions[partition_indices[member_index - 1]];
                        const auto &curr = partitions[partition_indices[member_index]];
                        if ((curr.rx_index != prev.rx_index + 1) or (curr.connection_begin != prev.connection_end)) {
                            contiguous = false;
                            break;
                        }
                    }
                    if (not contiguous) {
                        continue;
                    }
                    // add dense layer
                    const auto &first = partitions[partition_indices.front()];
                    _layers.push_back({.tx_begin     = connections[first.connection_begin].tx_index,
                                      .tx_count      = first.connection_end - first.connection_begin,
                                      .rx_begin      = first.rx_index,
                                      .rx_count      = std::ssize(partition_indices),
                                      .weight_offset = first.connection_begin});
                    for (const auto partition_index : partition_indices) {
                        _layer_indices[partition_index] = _layers.size() - 1;
                    }
                }
                std::println("DAGNet::init: found {} partitions(s)", partitions.size());
                std::println("DAGNet::init: found {} layer(s)", _layers.size());
            }
            // Propagate signals through the net
            void propagate_signals(State &state) const {
                const auto partitions = this->partitions();
                for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    const auto &partition = partitions[partition_index];
                    if (_layer_indices[partition_index] != Layer::no_layer) {
                        const auto &layer = _layers[_layer_indices[partition_index]];
                        propagate_layer(state, layer);
                        partition_index += layer.rx_count - 1;  // layer members are contiguous: skip them all at once
                    }
                    else {
                        propagate_partition(state, partition);
                    }
                }
            }
            // Propagate signals of a single partition
            void propagate_partition(State &state, const Partition &partition) const {
                real_t weighted_sum{0.0_real};
                const auto weights = this->weights();
                const auto connections = this->connections();
                const auto neurons = this->neurons();
                auto connection_index = partition.connection_begin;
                for (; connection_index <= (partition.connection_end - simd_register_size); connection_index += simd_register_size) {
                    weighted_sum += 
                        + weights[connection_index]     * state.signals[connections[connection_index].tx_index]
                        + weights[connection_index + 1] * state.signals[connections[connection_index + 1].tx_index]
                        + weights[connection_index + 2] * state.signals[connections[connection_index + 2].tx_index]
                        + weights[connection_index + 3] * state.signals[connections[connection_index + 3].tx_index];
                }
                //#pragma omp simd reduction(+:weighted_sum)
                for (; connection_index < partition.connection_end; connection_index++) {
                    weighted_sum += weights[connection_index] * state.signals[connections[connection_index].tx_index];
                }
                state.weighted_sums[partition.rx_index] = weighted_sum;
                state.signals[partition.rx_index] = neurons[partition.rx_index].activate(weighted_sum);
            }
            // Propagate signals of a layer
            void propagate_layer(State &state, const Layer &layer) const {
                for (auto row_index{0}; row_index < layer.rx_count; ++row_index) {
                    real_t weighted_sum{0.0_real};
                    const auto weights = this->weights();
                    const auto neurons = this->neurons();
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
                        weighted_sum +=  weights[row_offset + column_index] * state.signals[layer.tx_begin + column_index];
                    }
                    state.weighted_sums[layer.rx_begin + row_index] = weighted_sum;
                    state.signals[layer.rx_begin + row_index] = neurons[layer.rx_begin + row_index].activate(weighted_sum);
                }
            }
        private:
            // Data members
            std::vector<Layer> _layers{};
            index_vector_t _layer_indices{};
    };
}
