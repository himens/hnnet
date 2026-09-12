#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning_rule.h"

namespace hNNet {
    struct NNetState {
        std::vector<real_t> signals{};
        std::vector<real_t> weighted_sums{};
        NNetState() = default;
        explicit NNetState(const int_t neuron_count) : signals(neuron_count, 0.0), weighted_sums(neuron_count, 0.0) {}
    };
    ////////////////
    // NNet class //
    ////////////////
    template <DataType InputData, DataType OutputData>
        class NNet {
            public:
                // Aliases and constants
                using input_type = InputData;
                using output_type = OutputData;
                static constexpr size_t input_size{std::tuple_size_v<InputData>};
                static constexpr size_t output_size{std::tuple_size_v<OutputData>};
                // Data types
                struct TrainingData {
                    InputData  inputs;
                    OutputData targets;
                };
                struct SynapticConn {
                    index_t itx;
                    index_t irx;
                };
                struct Partition {
                    static constexpr index_t no_block{-1};
                    index_t irx;
                    index_t iconn_begin;
                    index_t iconn_end;
                    index_t iblock{no_block};
                    bool is_dense() const {
                        return iblock != no_block;
                    }
                };
                struct DenseBlock {
                    index_t itx_begin;
                    int_t tx_count;
                    index_t irx_begin;
                    int_t rx_count;
                    index_t weight_offset;
                };
                class View {
                    public:
                        using input_type = InputData;
                        using output_type = OutputData;
                        int_t neuron_count() const {
                            return _net._neurons.size();
                        }
                        int_t connection_count() const {
                            return _net._connections.size();
                        }
                        const Neuron& neuron(const index_t index) const {
                            return _net._neurons[index];
                        }
                        const SynapticConn& connection(const index_t index) const {
                            return _net._connections[index];
                        }
                        real_t& weight(const index_t index) {
                            return _net._weights[index];
                        }
                        const std::vector<Partition>& partitions() const {
                            return _net._partitions;
                        }
                        const NNet::DenseBlock& dense_block(const index_t index) const {
                            return _net._dense_blocks[index];
                        }
                        const std::vector<index_t>& iout_neurons() const {
                            return _net._iout_neurons;
                        }
                    private:
                        friend class NNet;
                        explicit View(NNet& net) : _net(net) {}
                        NNet& _net;
                };
            public:
                // Create a view of the net
                View view() {
                    return View(*this);
                }
                // Create new neurons
                template <ActivationType Activation>
                    std::vector<index_t> new_neurons(const int_t number, const NeuronType &type, const Activation &activation) {
                        if (number < 0) {
                            throw std::invalid_argument("NNet::new_neurons: invalid number!");
                        }
                        _trained = false;
                        for (auto i{0}; i < number; ++i) {
                            _neurons.push_back(Neuron{type, std::make_unique<Activation>(activation)});
                        }
                        return std::views::iota(_neurons.size() - number, _neurons.size()) | std::ranges::to<std::vector<index_t>>();
                }
                // Connect neurons (cartesian product)
                template <IndexRange TxRange, IndexRange RxRange>
                    void connect(const TxRange &itxs, const RxRange &irxs) {
                        _trained = false;
                        std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                        for (const auto &[itx, irx] : std::views::cartesian_product(itxs, irxs)) {
                            if ((itx < 0 or itx >= std::ssize(_neurons)) or (irx < 0 or irx >= std::ssize(_neurons))) {
                                throw std::out_of_range("NNet::connect: index out-of-range!");
                            }
                            const index_t iconn = std::ssize(_connections);
                            const auto [it, inserted] = hash_map.try_emplace(std::pair{irx, itx}, iconn);
                            if (not inserted) {
                                throw std::invalid_argument("NNet::connect: duplicate connection!");
                            }
                            _connections.push_back({.itx = itx, .irx = irx});
                            _weights.push_back(0.0);
                        }
                    }
                // Connect neurons (zip)
                template <IndexRange TxRange, IndexRange RxRange>
                    void zip_connect(const TxRange &itxs, const RxRange &irxs) {
                        if (itxs.size() != irxs.size()) {
                            throw std::invalid_argument("NNet::zip_connect: size error!");
                        }
                        for (const auto &[itx, irx] : std::views::zip(itxs, irxs)) {
                            connect(itx, irx);
                        }
                    }
                // Train net using a set of training samples
                template <typename LearningRule>
                    requires LearningRuleType<LearningRule, NNet>
                    void train(const std::span<TrainingData> samples, LearningRule rule) {
                        constexpr real_t loss_threshold{1e-2};
                        constexpr int_t max_epochs{1'000'000};
                        int_t epoch{0};
                        bool converged{false};
                        Timer timer{};
                        std::println("NNet::train: ==================================");
                        std::println("NNet::train: Training net with {} samples...   ", samples.size());
                        std::println("NNet::train: ==================================");
                        timer.start();
                        prepare();
                        while (not converged and (epoch <= max_epochs)) {
                            const auto loss = rule.learn(*this, samples);
                            converged = (loss < loss_threshold);
                            epoch++;
                            std::println("NNet::train: epoch: {}, elapsed time: {}s, loss: {:.6f}", epoch, timer.get_elapsed_time_s(), loss);
                        }
                        if (converged) {
                            _trained = true;
                            std::println("NNet::train: Training summary:");
                            std::println("NNet::train: elapsed time: {}s", timer.get_elapsed_time_s());
                            std::println("NNet::train: epochs: {}", epoch);
                        }
                        else {
                            _trained = false;
                            std::println("NNet::train: net training failed!");
                        }
                    }
                // Infer from data
                OutputData infer(const InputData &data) {
                    if (not _trained) {
                        std::println("NNet::infer: try to infer from an untrained net!");
                        return {};
                    }
                    NNetState state(_neurons.size());
                    update(state, data);
                    OutputData outputs;
                    for (const auto &[i, iout] : _iout_neurons | std::views::enumerate) {
                        outputs[i] = state.signals[iout];
                    }
                    return outputs;
                }
                // Update a net state from input data
                void update(NNetState &state, const InputData &inputs) const {
                    inject(state, inputs);
                    broadcast(state);
                }
            private:
                // Data types
                using IndexPair = std::pair<index_t, index_t>;
                struct IndexPairHash {
                    int_t operator()(const IndexPair &pair) const {
                        return std::hash<index_t>{}(pair.first) ^ (std::hash<index_t>{}(pair.second) << 1);
                    }
                };
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
                        std::vector<index_t> _roots;
                        std::vector<int_t> _ranks;
                };
                // Inject input data into a net state
                void inject(NNetState &state, const InputData &inputs) const {
                    for (const auto &[iin, input] : std::views::zip(_iin_neurons, inputs)) {
                        state.weighted_sums[iin] = input;
                        state.signals[iin] = _neurons[iin].activate(input);
                    }
                    for (const auto &ibias : _ibias_neurons) {
                        state.weighted_sums[ibias] = 1.0;
                        state.signals[ibias] = _neurons[ibias].activate(1.0);
                    }
                }
                // Broadcast signals through the net using the partitions, already in topological order
                void broadcast(NNetState &state) const {
                    for (auto ipart{0}; ipart < std::ssize(_partitions); ipart++) {
                        const auto &partition = _partitions[ipart];
                        if (partition.is_dense()) {
                            const auto iblock = partition.iblock;
                            const auto &block = _dense_blocks[iblock];
                            broadcast(block, state);
                            ipart += block.rx_count - 1;  // dense block members are contiguous: skip them all at once
                            continue;
                        }
                        broadcast(partition, state);
                    }
                }
                // Process a single partition
                void broadcast(const Partition &partition, NNetState &state) const {
                    real_t weighted_sum{0.0};
                    auto iconn = partition.iconn_begin;
                    for (; iconn <= (partition.iconn_end - register_size); iconn += register_size) {
                        weighted_sum +=  _weights[iconn]     * state.signals[_connections[iconn].itx]
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
                // Process a dense block: all its receivers share the exact same source range, hence a pure matrix-vector product
                void broadcast(const DenseBlock &block, NNetState &state) const {
                    for (auto irow{0}; irow < block.rx_count; ++irow) {
                        real_t weighted_sum{0.0};
                        const auto row_offset = block.weight_offset + irow * block.tx_count;
                        index_t icol{0};
                        for (; icol <= (block.tx_count - register_size); icol += register_size) {
                            weighted_sum +=  _weights[row_offset + icol]     * state.signals[block.itx_begin + icol]
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
                // Prepare net (build and order partitions, find dense blocks ...)
                void prepare() {
                    // sanity check 
                    _iout_neurons.clear();
                    _iin_neurons.clear();
                    _ibias_neurons.clear();
                    for (const auto &[inr, neuron] : _neurons | std::views::enumerate) {
                        switch(neuron.type()) {
                            case NeuronType::input:
                                _iin_neurons.push_back(inr);
                                break;
                            case NeuronType::output:
                                _iout_neurons.push_back(inr);
                                break;
                            case NeuronType::bias:
                                _ibias_neurons.push_back(inr);
                                break;
                           default:
                                break;
                        }
                    }
                    if ((_iin_neurons.size() != input_size) or (_iout_neurons.size() != output_size)) {
                        throw std::invalid_argument("NNet::prepare: size error!");
                    }
                    // randomize weights
                    std::uniform_real_distribution<real_t> dist(-0.1, +0.1);
                    for (auto &weight : _weights) {
                        weight = dist(random_generator());
                    }
                    // sort connections per irx and itx
                    std::ranges::sort(_connections, [&] (const auto &lhs, const auto &rhs) { return std::tie(lhs.irx, lhs.itx) < std::tie(rhs.irx, rhs.itx); });
                    // each contiguous block sharing the same irx becomes a partition
                    _partitions.clear();
                    index_t iconn_begin{0};
                    while (iconn_begin < std::ssize(_connections)) {
                        const auto &irx = _connections[iconn_begin].irx;
                        auto iconn_end = iconn_begin + 1;
                        while ((iconn_end < std::ssize(_connections)) and (_connections[iconn_end].irx == irx)) {
                            ++iconn_end;
                        }
                        _partitions.push_back({.irx = irx, .iconn_begin = iconn_begin, .iconn_end = iconn_end});
                        iconn_begin = iconn_end;
                    }
                    // topologically order partitions (Kahn's algorithm)
                    std::vector<std::vector<index_t>> irxs(_neurons.size());
                    for (const auto &conn : _connections) {
                        irxs[conn.itx].push_back(conn.irx);
                    }
                    std::vector<int_t> visits_left(_neurons.size(), 0);
                    for (const auto &partition : _partitions) {
                        visits_left[partition.irx] = partition.iconn_end - partition.iconn_begin;
                    }
                    std::vector<index_t> visited_queue;
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
                        throw std::runtime_error("NNet::prepare: net contains a cycle, topological order does not exist!");
                    }
                    // sort partitions per topological rank
                    std::ranges::sort(_partitions, [&] (const auto &lhs, const auto &rhs) { return topo_ranks[lhs.irx] < topo_ranks[rhs.irx]; });
                    // group partitions sharing the exact same (contiguous) tx range
                    _dense_blocks.clear();
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
                    std::unordered_map<index_t, std::vector<index_t>> groups;
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
                            _partitions[ipart].iblock = _dense_blocks.size() - 1;
                        }
                    }
                    std::println("NNet::prepare: neuron(s): {}, connection(s): {}", _neurons.size(), _connections.size());
                    std::println("NNet::prepare: found {} partitions(s)", _partitions.size());
                    std::println("NNet::prepare: found {} dense block(s)", _dense_blocks.size());
                }
                // Data members
                bool _trained{false};
                std::vector<Neuron> _neurons{};
                std::vector<SynapticConn> _connections{};
                std::vector<real_t> _weights{};
                std::vector<index_t> _iin_neurons{};
                std::vector<index_t> _iout_neurons{};
                std::vector<index_t> _ibias_neurons{};
                std::vector<Partition> _partitions{};
                std::vector<DenseBlock> _dense_blocks{};
        };
    template <typename T>
        using input_t = T::input_type;
    template <typename T>
        using output_t = T::output_type;
    template<typename T>
        constexpr auto input_size_v = T::input_size;
    template<typename T>
        constexpr auto output_size_v = T::output_size;
    template <typename T>
        concept NNetType = std::derived_from<T, NNet<input_t<T>, output_t<T>>>;
}
