#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning_rule.h"

namespace hNNet {
    using input_vector_t  = std::vector<real_t>;
    using output_vector_t = std::vector<real_t>;
    using index_vector_t  = std::vector<index_t>;
    struct NNetState {
        std::vector<real_t> signals{};
        std::vector<real_t> weighted_sums{};
        NNetState() = default;
        explicit NNetState(const int_t neuron_count) : signals(neuron_count, 0.0), weighted_sums(neuron_count, 0.0) {}
    };
    ////////////////
    // NNet class //
    ////////////////
    class NNet {
        public:
            // Data types
            struct SynapticConn {
                index_t itx;
                index_t irx;
            };
            struct Partition {
                index_t irx;
                index_t iconn_begin;
                index_t iconn_end;
            };
            virtual ~NNet() = default;
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
                    const SynapticConn& connection(const index_t index) const {
                        return _net._connections[index];
                    }
                    real_t& weight(const index_t index) {
                        return _net._weights[index];
                    }
                    const std::vector<Partition>& partitions() const {
                        return _net._partitions;
                    }
                    const index_vector_t& iout_neurons() const {
                        return _net._iout_neurons;
                    }
                protected:
                    explicit View(NNet& net) : _net(net) {}
                private:
                    friend class NNet;
                    NNet& _net;
            };
        public:
            // Create a view of the net
            View view() {
                return View(*this);
            }
            // Create new neurons
            template <ActivationType Activation>
                index_vector_t new_neurons(const int_t number, const NeuronType &type, const Activation &activation) {
                    if (number < 0) {
                        throw std::invalid_argument("NNet::new_neurons: invalid number!");
                    }
                    _trained = false;
                    for (auto i{0}; i < number; ++i) {
                        const auto inr = std::ssize(_neurons);
                        _neurons.push_back(Neuron{type, std::make_unique<Activation>(activation)});
                        switch (type) {
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
                    return std::views::iota(_neurons.size() - number, _neurons.size()) | std::ranges::to<index_vector_t>();
                }
            // Connect neurons (cartesian product)
            void connect(const IndexRange auto &itxs, const IndexRange auto &irxs) {
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
            void zip_connect(const IndexRange auto &itxs, const IndexRange auto &irxs) {
                if (itxs.size() != irxs.size()) {
                    throw std::invalid_argument("NNet::zip_connect: size error!");
                }
                for (const auto &[itx, irx] : std::views::zip(itxs, irxs)) {
                    connect(itx, irx);
                }
            }
            // Train net using a set of training samples
            template <typename Self, DatasetType Inputs, DatasetType Targets, typename LearningRule>
                void train(Self &self, const Inputs &inputs, const Targets &targets, LearningRule rule) {
                    if (std::ranges::size(inputs) != std::ranges::size(targets)) {
                        throw std::invalid_argument("NNet::train: size error!");
                    }
                    constexpr real_t loss_threshold{1e-2};
                    constexpr int_t max_epochs{1'000'000};
                    int_t epoch{0};
                    bool converged{false};
                    utils::Timer timer{};
                    std::println("NNet::train: ==================================");
                    std::println("NNet::train: Training net with {} samples...   ", std::ranges::size(inputs));
                    std::println("NNet::train: ==================================");
                    timer.start();
                    prepare();
                    for (const auto &[input, target] : std::views::zip(inputs, targets)) {
                        if (std::ssize(input) != std::ssize(_iin_neurons) or (std::ssize(target) != std::ssize(_iout_neurons))) {
                            throw std::invalid_argument("NNet::train: size error!");
                        }
                    }
                    while (not converged and (epoch <= max_epochs)) {
                        const auto loss = rule.learn(self, inputs, targets);
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
            output_vector_t infer(const DataType auto &inputs) {
                if (not _trained) {
                    std::println("NNet::infer: try to infer from an untrained net!");
                    return {};
                }
                if (std::ranges::size(inputs) != std::ranges::size(_iin_neurons)) {
                    throw std::invalid_argument("NNet::infer: size error!");
                }
                static thread_local NNetState state(_neurons.size());
                update(state, inputs);
                output_vector_t outputs(_iout_neurons.size());
                for (const auto &[i, iout] : _iout_neurons | std::views::enumerate) {
                    outputs[i] = state.signals[iout];
                }
                return outputs;
            }
            // Update a net state from input data
            void update(NNetState &state, const DataType auto &inputs) const {
                if (std::ssize(inputs) != std::ssize(_iin_neurons)) {
                    throw std::invalid_argument("NNet::update: size error!");
                }
                seed_state(state, inputs);
                update_state(state);
            }
        protected:
            // Data types
            using IndexPair = std::pair<index_t, index_t>;
            struct IndexPairHash {
                int_t operator()(const IndexPair &pair) const {
                    return std::hash<index_t>{}(pair.first) ^ (std::hash<index_t>{}(pair.second) << 1);
                }
            };
            // Sort connections per irx and itx
            static void sort_connections(const std::span<SynapticConn> connections) {
                std::ranges::sort(connections, [] (const auto &lhs, const auto &rhs) { return std::tie(lhs.irx, lhs.itx) < std::tie(rhs.irx, rhs.itx); });
            }
            // Group connections sharing the same rx into partitions
            static std::vector<Partition> make_partitions(const std::span<SynapticConn> connections) {
                std::vector<Partition> partitions;
                index_t iconn_begin{0};
                while (iconn_begin < std::ssize(connections)) {
                    const auto &irx = connections[iconn_begin].irx;
                    auto iconn_end = iconn_begin + 1;
                    while ((iconn_end < std::ssize(connections)) and (connections[iconn_end].irx == irx)) {
                        ++iconn_end;
                    }
                    partitions.push_back({.irx = irx, .iconn_begin = iconn_begin, .iconn_end = iconn_end});
                    iconn_begin = iconn_end;
                }
                return partitions;
            }
            // Seed a net state from input data
            void seed_state(NNetState &state, const DataType auto &inputs) const {
                for (const auto &[iin, input] : std::views::zip(_iin_neurons, inputs)) {
                    state.weighted_sums[iin] = input;
                    state.signals[iin] = _neurons[iin].activate(input);
                }
                for (const auto &ibias : _ibias_neurons) {
                    state.weighted_sums[ibias] = 1.0;
                    state.signals[ibias] = _neurons[ibias].activate(1.0);
                }
            }
            // Prepare net (default: randomize weights, sort connections, group them into partitions)
            virtual void prepare() {
                sort_connections(_connections);
                _partitions = make_partitions(_connections);
                _weights = utils::random::generate<real_t>(_weights.size(), -0.1, +0.1);
            }
            // Update net state
            virtual void update_state(NNetState &state) const = 0;
        protected:
            // Data members
            bool _trained{false};
            std::vector<Neuron> _neurons{};
            std::vector<SynapticConn> _connections{};
            std::vector<real_t> _weights{};
            std::vector<Partition> _partitions{};
            index_vector_t _iin_neurons{};
            index_vector_t _iout_neurons{};
            index_vector_t _ibias_neurons{};
    };
}
