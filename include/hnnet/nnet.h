#pragma once
#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hnnet/neuron.h"
#include "utils/random.h"
#include "utils/timer.h"

namespace hnnet {
    using input_vector_t  = std::vector<real_t>;
    using output_vector_t = std::vector<real_t>;
    using index_vector_t  = std::vector<index_t>; 
    ////////////////
    // NNet class //
    ////////////////
    class NNet {
        public:
            // Data types
            struct SynapticConn {
                index_t tx_index;
                index_t rx_index;
            };
            struct Partition {
                index_t rx_index;
                index_t connection_begin;
                index_t connection_end;
            };
            struct State {
                State() = default;
                explicit State(const int_t neuron_count) : signals(neuron_count, 0.0_real), weighted_sums(neuron_count, 0.0_real) {}
                std::vector<real_t> signals{};
                std::vector<real_t> weighted_sums{};
            };
            class View {
                public:
                    int_t neuron_count() const {
                        return _net._neurons.size();
                    }
                    int_t connection_count() const {
                        return _net._connections.size();
                    }
                    std::span<const Neuron> neurons() const {
                        return _net._neurons;
                    }
                    std::span<const SynapticConn> connections() const {
                        return _net._connections;
                    }
                    std::span<real_t> weights() const {
                        return _net._weights;
                    }
                    std::span<const Partition> partitions() const {
                        return _net._partitions;
                    }
                    std::span<const index_t> output_neurons_indices() const {
                        return _net._output_neurons_indices;
                    }
                protected:
                    explicit View(NNet& net) : _net(net) {}
                private:
                    friend class NNet;
                    NNet& _net;
            };
        public:
            // Destructor
            virtual ~NNet() = default;
            // Create a view of the net
            View view() {
                return View(*this);
            }
            // Train net using a set of training samples
            template <typename Self, DatasetType Inputs, DatasetType Targets, typename LearningRule>
                void train(Self &self, 
                           const Inputs &inputs, 
                           const Targets &targets, 
                           LearningRule rule) {
                    if (std::ranges::size(inputs) != std::ranges::size(targets)) {
                        throw std::invalid_argument("NNet::train: size error!");
                    }
                    constexpr int_t max_epochs{1'000'000};
                    int_t epoch{0};
                    utils::Timer timer{};
                    std::println("NNet::train: ==================================");
                    std::println("NNet::train: Training net with {} samples...   ", std::ranges::size(inputs));
                    std::println("NNet::train: ==================================");
                    timer.start();
                    _weights = utils::random::generate<real_t>(_weights.size(), -0.1_real, +0.1_real);
                    rule.init(self);
                    for (const auto &[input, target] : std::views::zip(inputs, targets)) {
                        if (std::ssize(input) != std::ssize(_input_neurons_indices) or (std::ssize(target) != std::ssize(_output_neurons_indices))) {
                            throw std::invalid_argument("NNet::train: size error!");
                        }
                    }
                    while (not _trained and (epoch <= max_epochs)) {
                        _trained = rule.learn(self, inputs, targets);
                        epoch++;
                        std::println("NNet::train: epoch: {}, elapsed time: {:.5f}s", epoch, timer.get_elapsed_time_s());
                    }
                    if (_trained) {
                        std::println("NNet::train: Training summary:");
                        std::println("NNet::train: elapsed time: {:.5f}s", timer.get_elapsed_time_s());
                        std::println("NNet::train: epochs: {}", epoch);
                    }
                    else {
                        std::println("NNet::train: net training failed!");
                    }
                }
            // Infer from data
            output_vector_t infer(const DataType auto &inputs) {
                if (not _trained) {
                    std::println("NNet::infer: try to infer from an untrained net!");
                    return {};
                }
                if (std::ranges::size(inputs) != std::ranges::size(_input_neurons_indices)) {
                    throw std::invalid_argument("NNet::infer: size error!");
                }
                static thread_local State state(_neurons.size());
                update(state, inputs);
                output_vector_t outputs(_output_neurons_indices.size());
                for (const auto &[index, neuron_index] : _output_neurons_indices | std::views::enumerate) {
                    outputs[index] = state.signals[neuron_index];
                }
                return outputs;
            }
            // Update a net state from input data
            void update(State &state, const DataType auto &inputs) const {
                if (std::ssize(inputs) != std::ssize(_input_neurons_indices)) {
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
            // Initialize common topology data after the derived network has constructed it
            virtual void init() {
                if (has_duplicate_connection(_connections)) {
                    throw std::invalid_argument("NNet::init: duplicate connection!");
                }
                _partitions = make_partitions(_connections);
                std::println("NNet::init: neuron(s): {}, connection(s): {}", _neurons.size(), _connections.size());
            }
            // Update net state
            virtual void update_state(State &state) const = 0;
            // Getters
            std::span<const Neuron> neurons() const {
                return _neurons;
            }
            std::span<const Partition> partitions() const {
                return _partitions;
            }
            std::span<const SynapticConn> connections() const {
                return _connections;
            }
            std::span<const real_t> weights() const {
                return _weights;
            }
            std::span<Partition> partitions() {
                return _partitions;
            }
            // Create new neurons
            template <ActivationType Activation>
                index_vector_t new_neurons(const int_t count, const NeuronType &type, const Activation &activation) {
                    if (count < 0) {
                        throw std::invalid_argument("NNet::new_neurons: invalid count!");
                    }
                    _trained = false;
                    for (auto count_index{0}; count_index < count; ++count_index) {
                        const auto new_neuron_index = std::ssize(_neurons);
                        _neurons.push_back(Neuron{type, std::make_unique<Activation>(activation)});
                        switch (type) {
                            case NeuronType::input:
                                _input_neurons_indices.push_back(new_neuron_index);
                                break;
                            case NeuronType::output:
                                _output_neurons_indices.push_back(new_neuron_index);
                                break;
                            case NeuronType::bias:
                                _bias_neurons_indices.push_back(new_neuron_index);
                                break;
                            default:
                                break;
                        }
                    }
                    return std::views::iota(_neurons.size() - count, _neurons.size()) | std::ranges::to<index_vector_t>();
                }
            // Connect neurons (cartesian product)
            void connect(IndexRange auto &&tx_indices, IndexRange auto &&rx_indices) {
                //for (const auto &[tx_index, rx_index] : std::views::cartesian_product(tx_indices, rx_indices))
                for (const auto &tx_index : tx_indices) {
                    for (const auto &rx_index : rx_indices) {
                        if ((tx_index < 0 or tx_index >= std::ssize(_neurons)) or (rx_index < 0 or rx_index >= std::ssize(_neurons))) {
                            throw std::out_of_range("NNet::connect: index out-of-range!");
                        }
                        _connections.push_back({.tx_index = tx_index, .rx_index = rx_index});
                        _weights.push_back(0.0_real);
                    }
                }
            }
            // Connect neurons (zip)
            void zip_connect(IndexRange auto &&tx_indices, IndexRange auto &&rx_indices) {
                auto tx_iterator = std::ranges::begin(tx_indices);
                auto rx_iterator = std::ranges::begin(rx_indices);
                while ((tx_iterator != std::ranges::end(tx_indices)) and (rx_iterator != std::ranges::end(rx_indices))) {
                    connect(std::views::single(*tx_iterator), std::views::single(*rx_iterator));
                    ++tx_iterator;
                    ++rx_iterator;
                }
                if ((tx_iterator != std::ranges::end(tx_indices)) or (rx_iterator != std::ranges::end(rx_indices))) {
                    throw std::invalid_argument("NNet::zip_connect: size error!");
                }
            }
            // Check for duplicate connections
            static bool has_duplicate_connection(const std::span<SynapticConn> connections) {
                std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                for (const auto &[index, connection] : connections | std::views::enumerate) {
                    const auto [_, inserted] = hash_map.try_emplace(std::pair{connection.rx_index, connection.tx_index}, index);
                    if (not inserted) {
                        return true;
                    }
                }
                return false;
            }
            // Group connections sharing the same rx into partitions
            static std::vector<Partition> make_partitions(const std::span<SynapticConn> connections) {
                std::ranges::sort(connections, [] (const auto &lhs, const auto &rhs) { return std::tie(lhs.rx_index, lhs.tx_index) < std::tie(rhs.rx_index, rhs.tx_index); });
                std::vector<Partition> partitions;
                index_t connection_begin{0};
                while (connection_begin < std::ssize(connections)) {
                    const auto &rx_index = connections[connection_begin].rx_index;
                    auto connection_end = connection_begin + 1;
                    while ((connection_end < std::ssize(connections)) and (connections[connection_end].rx_index == rx_index)) {
                        ++connection_end;
                    }
                    partitions.push_back({.rx_index = rx_index, .connection_begin = connection_begin, .connection_end = connection_end});
                    connection_begin = connection_end;
                }
                return partitions;
            }
        private:
            // Seed a net state from input data
            void seed_state(State &state, const DataType auto &inputs) const {
                for (const auto &[neuron_index, input] : std::views::zip(_input_neurons_indices, inputs)) {
                    state.weighted_sums[neuron_index] = input;
                    state.signals[neuron_index] = _neurons[neuron_index].activate(input);
                }
                for (const auto &neuron_index : _bias_neurons_indices) {
                    state.weighted_sums[neuron_index] = 1.0_real;
                    state.signals[neuron_index] = _neurons[neuron_index].activate(1.0_real);
                }
            }
        private:
            // Data members
            bool _trained{false};
            std::vector<Neuron> _neurons{};
            std::vector<SynapticConn> _connections{};
            std::vector<real_t> _weights{};
            std::vector<Partition> _partitions{};
            index_vector_t _input_neurons_indices{};
            index_vector_t _output_neurons_indices{};
            index_vector_t _bias_neurons_indices{};
    };
}
