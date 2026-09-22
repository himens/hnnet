#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    ///////////////////
    // DAGNet class  //
    ///////////////////
    class DAGNet : public NNet {
        public:
            // Data types
            struct DenseBlock {
                static constexpr index_t no_block{-1};
                index_t itx_begin;
                int_t tx_count;
                index_t irx_begin;
                int_t rx_count;
                index_t weight_offset;
            };
            class View : public NNet::View {
                public:
                    bool is_dense(const index_t ipart) const {
                        return _net._iblocks[ipart] != DenseBlock::no_block;
                    }
                    const DenseBlock& dense_block(const index_t ipart) const {
                        return _net._dense_blocks[_net._iblocks[ipart]];
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
                    NNet::train(*this, inputs, targets, std::move(rule));
                }
        protected:
            // Update net state
            void update_state(NNetState &state) const override {
                propagate_signals(state);
            }
            // Prepare net (order partitions topologically, find dense blocks...)
            void prepare() override {
                NNet::prepare();
                // topologically order partitions (Kahn's algorithm)
                std::vector<index_vector_t> irxs(_neurons.size());
                for (const auto &conn : _connections) {
                    irxs[conn.itx].push_back(conn.irx);
                }
                std::vector<int_t> visits_left(_neurons.size(), 0);
                for (const auto &partition : _partitions) {
                    visits_left[partition.irx] = partition.iconn_end - partition.iconn_begin;
                }
                index_vector_t visited_queue;
                for (auto inr{0}; inr < std::ssize(_neurons); ++inr) {
                    if (visits_left[inr] == 0) {
                        visited_queue.push_back(inr);
                    }
                }
                int_t visited_count{0};
                std::vector<int_t> topo_ranks(_neurons.size(), 0);
                while (not visited_queue.empty()) {
                    const auto inr = visited_queue.back();
                    visited_queue.pop_back();
                    topo_ranks[inr] = visited_count++;
                    for (const auto &irx : irxs[inr]) {
                        if (--visits_left[irx] == 0) {
                            visited_queue.push_back(irx);
                        }
                    }
                }
                if (visited_count != std::ssize(_neurons)) {
                    throw std::runtime_error("DAGNet::prepare: net contains a cycle, topological order does not exist!");
                }
                // sort partitions per topological rank
                std::ranges::sort(_partitions, [&] (const auto &lhs, const auto &rhs) { return topo_ranks[lhs.irx] < topo_ranks[rhs.irx]; });
                // group partitions sharing the exact same (contiguous) tx range
                _dense_blocks.clear();
                _iblocks.assign(std::ssize(_partitions), DenseBlock::no_block);
                UnionFind union_find(_partitions.size());
                std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                for (const auto &[ipart, partition] : _partitions | std::views::enumerate) {
                    const auto count = partition.iconn_end - partition.iconn_begin;
                    const auto itx_begin = _connections[partition.iconn_begin].itx;
                    if ((_connections[partition.iconn_end - 1].itx - itx_begin) != (count - 1)) {
                        continue;  // itx range has gaps (e.g. a bias mixed in): not a pure dense candidate
                    }
                    const auto [it, inserted] = hash_map.try_emplace(std::pair{itx_begin, count}, ipart);
                    if (not inserted) {
                        union_find.unite(it->second, ipart);
                    }
                }
                std::unordered_map<index_t, index_vector_t> groups;
                for (auto ipart{0}; ipart < std::ssize(_partitions); ++ipart) {
                    groups[union_find.find(ipart)].push_back(ipart);
                }
                // find dense blocks
                for (auto &[root, members] : groups) {
                    if (members.size() < 2) {
                        continue;  // no gain grouping a single receiver
                    }
                    const auto [min_ipart, max_ipart] = std::ranges::minmax(members);
                    if ((max_ipart - min_ipart + 1) != std::ssize(members)) {
                        continue; // must be contiguous partitions (in topo order)
                    }
                    // sort members per increasing irx and check their irxs and itxs are contiguous
                    std::ranges::sort(members, [&] (const auto &lhs, const auto &rhs) { return _partitions[lhs].irx < _partitions[rhs].irx; });
                    auto contiguous = true;
                    for (auto i{1}; i < std::ssize(members); ++i) {
                        const auto &prev = _partitions[members[i - 1]];
                        const auto &curr = _partitions[members[i]];
                        if ((curr.irx != prev.irx + 1) or (curr.iconn_begin != prev.iconn_end)) {
                            contiguous = false;
                            break;
                        }
                    }
                    if (not contiguous) {
                        continue;
                    }
                    // add dense block
                    const auto &first = _partitions[members.front()];
                    _dense_blocks.push_back({
                        .itx_begin = _connections[first.iconn_begin].itx,
                        .tx_count = first.iconn_end - first.iconn_begin,
                        .irx_begin = first.irx,
                        .rx_count = std::ssize(members),
                        .weight_offset = first.iconn_begin,
                    });
                    for (auto &ipart : members) {
                        _iblocks[ipart] = _dense_blocks.size() - 1;
                    }
                }
                std::println("DAGNet::prepare: neuron(s): {}, connection(s): {}", _neurons.size(), _connections.size());
                std::println("DAGNet::prepare: found {} partitions(s)", _partitions.size());
                std::println("DAGNet::prepare: found {} dense block(s)", _dense_blocks.size());
            }
        private:
            // Data types
            class UnionFind {
                public:
                    explicit UnionFind(const int_t count) : _roots(count), _ranks(count, 0) {
                        if (count < 0) {
                            throw std::invalid_argument("UnionFind: invalid count!");
                        }
                        std::iota(std::begin(_roots), std::end(_roots), 0);
                    }
                    index_t find(const index_t x) {
                        if (_roots[x] == x) {
                            return x;
                        }
                        return _roots[x] = find(_roots[x]);
                    }
                    void unite(const index_t a, const index_t b) {
                        auto root_a = find(a);
                        auto root_b = find(b);
                        if (root_a == root_b) {
                            return;
                        }
                        if (_ranks[root_a] < _ranks[root_b]) {
                            std::swap(root_a, root_b);
                        }
                        _roots[root_b] = root_a;
                        if (_ranks[root_a] == _ranks[root_b]) {
                            _ranks[root_a]++;
                        }
                    }
                private:
                    index_vector_t _roots;
                    std::vector<int_t> _ranks;
            };
            // Propagate signals through the net
            void propagate_signals(NNetState &state) const {
                for (auto ipart{0}; ipart < std::ssize(_partitions); ++ipart) {
                    const auto &partition = _partitions[ipart];
                    if (_iblocks[ipart] != DenseBlock::no_block) {
                        const auto &block = _dense_blocks[_iblocks[ipart]];
                        propagate_dense_block(state, block);
                        ipart += block.rx_count - 1;  // dense block members are contiguous: skip them all at once
                        continue;
                    }
                    propagate_partition(state, partition);
                }
            }
            // Propagate signals in a single partition
            void propagate_partition(NNetState &state, const Partition &partition) const {
                real_t weighted_sum{0.0};
                auto iconn = partition.iconn_begin;
                for (; iconn <= (partition.iconn_end - register_size); iconn += register_size) {
                    weighted_sum +=   _weights[iconn]     * state.signals[_connections[iconn].itx]
                                    + _weights[iconn + 1] * state.signals[_connections[iconn + 1].itx]
                                    + _weights[iconn + 2] * state.signals[_connections[iconn + 2].itx]
                                    + _weights[iconn + 3] * state.signals[_connections[iconn + 3].itx];
                }
                //#pragma omp simd reduction(+:weighted_sum)
                for (; iconn < partition.iconn_end; iconn++) {
                    weighted_sum += _weights[iconn] * state.signals[_connections[iconn].itx];
                }
                state.weighted_sums[partition.irx] = weighted_sum;
                state.signals[partition.irx] = _neurons[partition.irx].activate(weighted_sum);
            }
            // Propagate signals in dense block
            void propagate_dense_block(NNetState &state, const DenseBlock &block) const {
                for (auto irow{0}; irow < block.rx_count; ++irow) {
                    real_t weighted_sum{0.0};
                    const auto row_offset = block.weight_offset + irow * block.tx_count;
                    index_t icol{0};
                    for (; icol <= (block.tx_count - register_size); icol += register_size) {
                        weighted_sum +=   _weights[row_offset + icol]     * state.signals[block.itx_begin + icol]
                                        + _weights[row_offset + icol + 1] * state.signals[block.itx_begin + icol + 1]
                                        + _weights[row_offset + icol + 2] * state.signals[block.itx_begin + icol + 2]
                                        + _weights[row_offset + icol + 3] * state.signals[block.itx_begin + icol + 3];
                    }
                    //#pragma omp simd reduction(+:weighted_sum)
                    for (; icol < block.tx_count; ++icol) {
                        weighted_sum +=  _weights[row_offset + icol] * state.signals[block.itx_begin + icol];
                    }
                    state.weighted_sums[block.irx_begin + irow] = weighted_sum;
                    state.signals[block.irx_begin + irow] = _neurons[block.irx_begin + irow].activate(weighted_sum);
                }
            }
            // Data members
            std::vector<DenseBlock> _dense_blocks{};
            index_vector_t _iblocks{};
    };
}
