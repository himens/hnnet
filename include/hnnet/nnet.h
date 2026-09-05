#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning-rule.h"

namespace hNNet {
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
                static constexpr index_t no_block{-1};
                // Data types
                struct TrainingSample {
                    InputData inputs;
                    OutputData targets;
                };
                struct SynapticConn {
                    index_t itx;
                    index_t irx;
                };
                struct Partition {
                    index_t irx;
                    index_t icon_begin;
                    index_t icon_end;
                    index_t iblock;
                };
                struct DenseBlock {
                    index_t itx_begin;
                    int_t tx_count;
                    index_t irx_begin;
                    int_t rx_count;
                    index_t weight_offset;
                };
                ////////////////
                // View class //
                ////////////////
                class View {
                    public:
                        int_t neuron_count() const {
                            return _net._neurons.size();
                        }
                        int_t connection_count() const {
                            return _net._connections.size();
                        }
                        const Neuron& neuron(const index_t index) const {
                            return _net._neurons[index];
                        }
                        const SynapticConn& connection(const index_t icon) const {
                            return _net._connections[icon];
                        }
                        real_t& weight(const index_t icon) {
                            return _net._weights[icon];
                        }
                        const std::vector<Partition>& partitions() const {
                            return _net._partitions;
                        }
                        const NNet::DenseBlock& dense_block(const index_t index) const {
                            return _net._dense_blocks[index];
                        }
                        real_t signal(const index_t index) const {
                            return _net._signals[index];
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
                // Constructor
                NNet() {
                    _neurons.reserve(1000);
                    _signals.reserve(_neurons.capacity());
                    _connections.reserve(_neurons.capacity() * 10);
                    _weights.reserve(_neurons.capacity() * 10);
                }
                // Create a view of the net
                View view() {
                    return View(*this);
                }
                // Create new neurons
                template <ActivationType Activation>
                    auto new_neurons(const int_t number, const NeuronType &type, const Activation &activation) {
                        if (number < 0) {
                            throw std::invalid_argument("NNet::new_neurons: invalid number!");
                        }
                        _trained = false;
                        for (auto i{0}; i < number; ++i) {
                            _neurons.push_back({type, std::make_unique<Activation>(activation)});
                            _signals.push_back(0.0);
                        }
                        return _neurons | std::views::drop(_neurons.size() - number) | std::views::take(number);
                }
                // Connect neurons (cartesian product)
                template <typename TxType, typename RxType>
                    requires (NeuronView<std::remove_cvref_t<TxType>> or std::same_as<std::remove_cvref_t<TxType>, Neuron>) and
                             (NeuronView<std::remove_cvref_t<RxType>> or std::same_as<std::remove_cvref_t<RxType>, Neuron>)
                    void connect(TxType &&tx_neurons, RxType &&rx_neurons) {
                        // Compute the flat index of neurons
                        auto index_of = [&] (const Neuron &neuron) {
                            const index_t index = &neuron - _neurons.data();
                            if (index < 0 or index >= std::ssize(_neurons)) {
                                throw std::invalid_argument("NNet::connect::index_of: invalid index!");
                            }
                            return index;
                        };
                        auto to_index = [&] (auto &&arg) {
                            if constexpr (std::same_as<std::remove_cvref_t<decltype(arg)>, Neuron>) {
                                return std::views::single(index_of(arg));
                            }
                            else {
                                return std::forward<decltype(arg)>(arg) | std::views::transform([&] (const auto &neuron) { return index_of(neuron); });
                            }
                        };
                        _trained = false;
                        auto itxs = to_index(std::forward<TxType>(tx_neurons));
                        auto irxs = to_index(std::forward<RxType>(rx_neurons));
                        std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                        for (const auto &[itx, irx] : std::views::cartesian_product(itxs, irxs)) {
                            const index_t icon = std::ssize(_connections);
                            const auto [it, inserted] = hash_map.try_emplace(std::pair{irx, itx}, icon);
                            if (not inserted) {
                                throw std::invalid_argument("NNet::connect: duplicate connection!");
                            }
                            _connections.push_back({.itx = itx, .irx = irx});
                            _weights.push_back(0.0);
                        }
                    }
                // Connect neurons (zip)
                template <NeuronView TxView, NeuronView RxView>
                    void zip_connect(TxView itxs, RxView irxs) {
                        for (const auto &[tx, rx] : std::views::zip(itxs, irxs)) {
                            connect(tx, rx);
                        }
                    }
                // Train net using a set of training samples
                template <typename LearningRule>
                    requires LearningRuleType<LearningRule, NNet>
                    void train(const std::vector<TrainingSample> &samples, LearningRule rule) {
                        constexpr real_t error_threshold{1e-2};
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
                            //std::ranges::shunion_findfle(samples, random_generator()); -- samples must be not const!
                            real_t mean_squared_error{0.0};
                            for (const auto &sample : samples) {
                                reset();
                                inject(sample.inputs);
                                broadcast();
                                mean_squared_error += learn(sample.targets, rule);
                            }
                            mean_squared_error /= samples.size();
                            converged = (mean_squared_error < error_threshold);
                            epoch++;
                            std::println("NNet::train: epoch: {}, elapsed time: {}s, error: {:.6f}", epoch, timer.get_elapsed_time_s(), mean_squared_error);
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
                    reset();
                    inject(data);
                    broadcast();
                    OutputData outputs;
                    for (const auto &[i, iout] : _iout_neurons | std::views::enumerate) {
                        outputs[i] = _signals[iout];
                    }
                    return outputs;
                }
            private:
                /////////////////////
                // UnionFind class //
                /////////////////////
                class UnionFind {
                    public:
                        explicit UnionFind(const int_t count) : _roots(count), _ranks(count, 0) {
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
                // Index pair hash
                using IndexPair = std::pair<index_t, index_t>;
                struct IndexPairHash {
                    int_t operator()(const IndexPair &pair) const {
                        return std::hash<index_t>{}(pair.first) ^ (std::hash<index_t>{}(pair.second) << 1);
                    }
                };
                // Inject input data into neurons
                void inject(const InputData &inputs) {
                    for (const auto &[iin, input] : std::views::zip(_iin_neurons, inputs)) {
                        _signals[iin] = _neurons[iin].activate(input);
                    }
                    for (const auto &ibias : _ibias_neurons) {
                        _signals[ibias] = _neurons[ibias].activate(1.0);
                    }
                } 
                // Broadcast signals through the net using the partitions, already in topological order
                void broadcast() {
                    for (auto ipart{0}; ipart < std::ssize(_partitions); ipart++) {
                        const auto &partition = _partitions[ipart];
                        const auto iblock = partition.iblock;;
                        if (iblock != no_block) {
                            const auto &block = _dense_blocks[iblock];
                            broadcast(block);
                            ipart += block.rx_count - 1;  // dense block members are contiguous: skip them all at once
                            continue;
                        }
                        broadcast(partition);
                    }
                }
                // Process a single partition
                void broadcast(const Partition &partition) {
                    real_t weighted_sum{0.0};
                    auto icon = partition.icon_begin;
                    for (; icon <= (partition.icon_end - register_size); icon += register_size) {
                        weighted_sum +=  _weights[icon]     * _signals[_connections[icon].itx]
                                       + _weights[icon + 1] * _signals[_connections[icon + 1].itx]
                                       + _weights[icon + 2] * _signals[_connections[icon + 2].itx]
                                       + _weights[icon + 3] * _signals[_connections[icon + 3].itx];
                    }
                    //#pragma omp simd reduction(+:weighted_sum)
                    for (; icon < partition.icon_end; icon++) {
                        weighted_sum += _weights[icon] * _signals[_connections[icon].itx];
                    }
                    _signals[partition.irx] = _neurons[partition.irx].activate(weighted_sum);
                }
                // Process a dense block: all its receivers share the exact same source range, hence a pure matrix-vector product
                void broadcast(const DenseBlock &block) {
                    for (auto irow{0}; irow < block.rx_count; ++irow) {
                        real_t weighted_sum{0.0};
                        const auto row_offset = block.weight_offset + irow * block.tx_count;
                        index_t icol{0};
                        for (; icol <= (block.tx_count - register_size); icol += register_size) {
                            weighted_sum +=  _weights[row_offset + icol]     * _signals[block.itx_begin + icol]
                                           + _weights[row_offset + icol + 1] * _signals[block.itx_begin + icol + 1]
                                           + _weights[row_offset + icol + 2] * _signals[block.itx_begin + icol + 2]
                                           + _weights[row_offset + icol + 3] * _signals[block.itx_begin + icol + 3];
                        }
                        //#pragma omp simd reduction(+:weighted_sum)
                        for (; icol < block.tx_count; ++icol) {
                            weighted_sum +=  _weights[row_offset + icol] * _signals[block.itx_begin + icol];
                        }
                        _signals[block.irx_begin + irow] = _neurons[block.irx_begin + irow].activate(weighted_sum);
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
                    index_t icon_begin{0};
                    while (icon_begin < std::ssize(_connections)) {
                        const auto &irx = _connections[icon_begin].irx;
                        auto icon_end = icon_begin + 1;
                        while ((icon_end < std::ssize(_connections)) and (_connections[icon_end].irx == irx)) {
                            ++icon_end;
                        }
                        _partitions.push_back({.irx = irx, .icon_begin = icon_begin, .icon_end = icon_end, .iblock = no_block});
                        icon_begin = icon_end;
                    }
                    // topologically order partitions (Kahn's algorithm)
                    std::vector<std::vector<index_t>> irxs(_neurons.size());
                    for (const auto &conn : _connections) {
                        irxs[conn.itx].push_back(conn.irx);
                    }
                    std::vector<int_t> visits_left(_neurons.size(), 0);
                    for (const auto &partition : _partitions) {
                        visits_left[partition.irx] = partition.icon_end - partition.icon_begin;
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
                        const auto count = partition.icon_end - partition.icon_begin;
                        const auto itx_begin = _connections[partition.icon_begin].itx;
                        if ((_connections[partition.icon_end - 1].itx - itx_begin) != (count - 1)) {
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
                            if ((curr.irx != prev.irx + 1) or (curr.icon_begin != prev.icon_end)) {
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
                            .itx_begin = _connections[first.icon_begin].itx,
                            .tx_count = first.icon_end - first.icon_begin,
                            .irx_begin = first.irx,
                            .rx_count = std::ssize(members),
                            .weight_offset = first.icon_begin,
                        });
                        for (auto &ipart : members) {
                            _partitions[ipart].iblock = _dense_blocks.size() - 1;
                        }
                    }
                    std::println("NNet::prepare: neuron(s): {}, connection(s): {}", _neurons.size(), _connections.size());
                    std::println("NNet::prepare: found {} partitions(s)", _partitions.size());
                    std::println("NNet::prepare: found {} dense block(s)", _dense_blocks.size());
                }
                // Learn from targets using the selected learning rule
                template <typename LearningRule>
                    requires LearningRuleType<LearningRule, NNet>
                    [[nodiscard]] real_t learn(const OutputData &targets, LearningRule &rule) {
                        return rule.learn(*this, targets);
                    }
                // Reset all neurons in the net
                void reset() {
                    for (const auto &[inr, neuron] : _neurons | std::views::enumerate) {
                        neuron.reset();
                        _signals[inr] = 0.0;
                    }
                }
                // Get random generator
                std::mt19937& random_generator() {
                    static std::random_device rd;
                    static thread_local std::mt19937 gen(rd());
                    return gen;
                }
                // Data members
                bool _trained{false};
                std::vector<Neuron> _neurons{};
                std::vector<real_t> _signals{};
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
        constexpr std::size_t input_size_v = T::input_size;
    template<typename T>
        constexpr std::size_t output_size_v = T::output_size;
    template <typename T>
        concept NNetType = std::derived_from<T, NNet<input_t<T>, output_t<T>>>;
}
