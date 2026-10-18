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
            struct DenseBlock {
                static constexpr index_t no_block{-1};
                index_t tx_begin;
                int_t tx_count;
                index_t rx_begin;
                int_t rx_count;
                index_t weight_offset;
            };
            class View : public NNet::View {
                public:
                    bool is_dense(index_t partition_index) const {
                        return _net._partition_dense_block_indices[partition_index] != DenseBlock::no_block;
                    }
                    const DenseBlock& dense_block(const index_t partition_index) const {
                        return _net._dense_blocks[_net._partition_dense_block_indices[partition_index]];
                    }
                private:
                    friend class DAGNet;
                    explicit View(DAGNet &net) : NNet::View(net), _net(net) {}
                    DAGNet &_net;
            };
        public:
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
            // Constructor
            DAGNet() = default;
            // Update net state
            void update_state(State &state) const override {
                propagate_signals(state);
            }
            // Initialize net
            void init() override final {
                NNet::init();
                auto connections = this->connections();
                auto partitions = this->partitions();
                auto neurons = this->neurons();
                // topologically order partitions (Kahn's algorithm)
                std::vector<index_vector_t> rx_neurons_indices(neurons.size());
                for (const auto &connection : connections) {
                    rx_neurons_indices[connection.tx_neuron_index].push_back(connection.rx_neuron_index);
                }
                std::vector<int_t> visits_left(neurons.size(), 0);
                for (const auto &partition : partitions) {
                    visits_left[partition.rx_neuron_index] = partition.conn_end - partition.conn_begin;
                }
                index_vector_t visited_neuron_indices;
                for (auto neuron_index{0}; neuron_index < std::ssize(neurons); ++neuron_index) {
                    if (visits_left[neuron_index] == 0) {
                        visited_neuron_indices.push_back(neuron_index);
                    }
                }
                int_t visited_count{0};
                std::vector<int_t> topo_ranks(neurons.size(), 0);
                while (not visited_neuron_indices.empty()) {
                    const auto neuron_index = visited_neuron_indices.back();
                    visited_neuron_indices.pop_back();
                    topo_ranks[neuron_index] = visited_count++;
                    for (const auto &rx_neuron_index : rx_neurons_indices[neuron_index]) {
                        if (--visits_left[rx_neuron_index] == 0) {
                            visited_neuron_indices.push_back(rx_neuron_index);
                        }
                    }
                }
                if (visited_count != std::ssize(neurons)) {
                    throw std::runtime_error("DAGNet::init: net contains a cycle, topological order does not exist!");
                }
                // sort partitions per topological rank
                std::ranges::sort(partitions, [&] (const auto &lhs, const auto &rhs) { return topo_ranks[lhs.rx_neuron_index] < topo_ranks[rhs.rx_neuron_index]; });
                // group partitions sharing the exact same (contiguous) tx range
                _partition_dense_block_indices.assign(std::ssize(partitions), DenseBlock::no_block);
                UnionFind union_find(partitions.size());
                std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                for (const auto &[partition_index, partition] : partitions | std::views::enumerate) {
                    const auto count = partition.conn_end - partition.conn_begin;
                    const auto tx_begin = connections[partition.conn_begin].tx_neuron_index;
                    if ((connections[partition.conn_end - 1].tx_neuron_index - tx_begin) != (count - 1)) {
                        continue;  // transmitter range has gaps (e.g. a bias mixed in): not a pure dense candidate
                    }
                    const auto [it, inserted] = hash_map.try_emplace(std::pair{tx_begin, count}, partition_index);
                    if (not inserted) {
                        union_find.unite(it->second, partition_index);
                    }
                }
                std::unordered_map<index_t, index_vector_t> groups;
                for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    groups[union_find.find(partition_index)].push_back(partition_index);
                }
                // find dense blocks
                _dense_blocks.clear();
                for (auto &[root_index, partition_indices] : groups) {
                    if (partition_indices.size() < 2) {
                        continue;  // no gain grouping a single receiver
                    }
                    const auto [min_partition_index, max_partition_index] = std::ranges::minmax(partition_indices);
                    if ((max_partition_index - min_partition_index + 1) != std::ssize(partition_indices)) {
                        continue; // must be contiguous partitions (in topo order)
                    }
                    // sort partition indices per increasing receiver and check their ranges are contiguous
                    std::ranges::sort(partition_indices, [&] (const auto &lhs, const auto &rhs) { return partitions[lhs].rx_neuron_index < partitions[rhs].rx_neuron_index; });
                    auto contiguous = true;
                    for (auto member_index{1}; member_index < std::ssize(partition_indices); ++member_index) {
                        const auto &prev = partitions[partition_indices[member_index - 1]];
                        const auto &curr = partitions[partition_indices[member_index]];
                        if ((curr.rx_neuron_index != prev.rx_neuron_index + 1) or (curr.conn_begin != prev.conn_end)) {
                            contiguous = false;
                            break;
                        }
                    }
                    if (not contiguous) {
                        continue;
                    }
                    // add dense block
                    const auto &first = partitions[partition_indices.front()];
                    _dense_blocks.push_back({.tx_begin     = connections[first.conn_begin].tx_neuron_index,
                                            .tx_count      = first.conn_end - first.conn_begin,
                                            .rx_begin      = first.rx_neuron_index,
                                            .rx_count      = std::ssize(partition_indices),
                                            .weight_offset = first.conn_begin});
                    for (const auto partition_index : partition_indices) {
                        _partition_dense_block_indices[partition_index] = _dense_blocks.size() - 1;
                    }
                }
                std::println("DAGNet::prepare: found {} partitions(s)", partitions.size());
                std::println("DAGNet::prepare: found {} dense block(s)", _dense_blocks.size());
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
                    index_t find(const index_t element_index) {
                        if (_root_indices[element_index] == element_index) {
                            return element_index;
                        }
                        return _root_indices[element_index] = find(_root_indices[element_index]);
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
            // Propagate signals through the net
            void propagate_signals(State &state) const {
                const auto partitions = this->partitions();
                for (auto partition_index{0}; partition_index < std::ssize(partitions); ++partition_index) {
                    const auto &partition = partitions[partition_index];
                    if (_partition_dense_block_indices[partition_index] != DenseBlock::no_block) {
                        const auto &block = _dense_blocks[_partition_dense_block_indices[partition_index]];
                        propagate_dense_block(state, block);
                        partition_index += block.rx_count - 1;  // dense block members are contiguous: skip them all at once
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
                auto connection_index = partition.conn_begin;
                for (; connection_index <= (partition.conn_end - simd_register_size); connection_index += simd_register_size) {
                    weighted_sum += 
                        + weights[connection_index]     * state.signals[connections[connection_index].tx_neuron_index]
                        + weights[connection_index + 1] * state.signals[connections[connection_index + 1].tx_neuron_index]
                        + weights[connection_index + 2] * state.signals[connections[connection_index + 2].tx_neuron_index]
                        + weights[connection_index + 3] * state.signals[connections[connection_index + 3].tx_neuron_index];
                }
                //#pragma omp simd reduction(+:weighted_sum)
                for (; connection_index < partition.conn_end; connection_index++) {
                    weighted_sum += weights[connection_index] * state.signals[connections[connection_index].tx_neuron_index];
                }
                state.weighted_sums[partition.rx_neuron_index] = weighted_sum;
                state.signals[partition.rx_neuron_index] = neurons[partition.rx_neuron_index].activate(weighted_sum);
            }
            // Propagate signals of a dense block
            void propagate_dense_block(State &state, const DenseBlock &block) const {
                for (auto row_index{0}; row_index < block.rx_count; ++row_index) {
                    real_t weighted_sum{0.0_real};
                    const auto weights = this->weights();
                    const auto neurons = this->neurons();
                    const auto row_offset = block.weight_offset + row_index * block.tx_count;
                    index_t column_index{0};
                    for (; column_index <= (block.tx_count - simd_register_size); column_index += simd_register_size) {
                        weighted_sum +=   
                            + weights[row_offset + column_index]     * state.signals[block.tx_begin + column_index]
                            + weights[row_offset + column_index + 1] * state.signals[block.tx_begin + column_index + 1]
                            + weights[row_offset + column_index + 2] * state.signals[block.tx_begin + column_index + 2]
                            + weights[row_offset + column_index + 3] * state.signals[block.tx_begin + column_index + 3];
                    }
                    //#pragma omp simd reduction(+:weighted_sum)
                    for (; column_index < block.tx_count; ++column_index) {
                        weighted_sum +=  weights[row_offset + column_index] * state.signals[block.tx_begin + column_index];
                    }
                    state.weighted_sums[block.rx_begin + row_index] = weighted_sum;
                    state.signals[block.rx_begin + row_index] = neurons[block.rx_begin + row_index].activate(weighted_sum);
                }
            }
        private:
            // Data members
            std::vector<DenseBlock> _dense_blocks{};
            index_vector_t _partition_dense_block_indices{};
    };
}
