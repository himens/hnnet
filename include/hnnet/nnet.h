#pragma once
#include <algorithm>
#include <array>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hnnet/topology.h"
#include "hnnet/learning_rule_type.h"
#include "hnnet/update_backend_type.h"
#include "utils/random.h"
#include "utils/timer.h"

namespace hnnet {
    ////////////////
    // NNet class //
    ////////////////
    class NNet {
        public:
            // Data types
            struct Partition {
                index_t rx_index;
                index_t connection_begin;
                index_t connection_end;
            };
            class View {
                public:
                    int_t neuron_count() const {
                        return std::ssize(_net->_neurons);
                    }
                    int_t input_neuron_count() const {
                        return std::ssize(_net->_input_neurons_indices);
                    }
                    int_t output_neuron_count() const {
                        return std::ssize(_net->_output_neurons_indices);
                    }
                    int_t connection_count() const {
                        return std::ssize(_net->_connections);
                    }
                    std::span<const Neuron> neurons() const {
                        return _net->_neurons;
                    }
                    std::span<const Connection> connections() const {
                        return _net->_connections;
                    }
                    std::span<const real_t> weights() const {
                        return _net->_weights;
                    }
                    std::span<const Partition> partitions() const {
                        return _net->_partitions;
                    }
                    std::span<const index_t> output_neurons_indices() const {
                        return _net->_output_neurons_indices;
                    }
                    std::span<const index_t> input_neurons_indices() const {
                        return _net->_input_neurons_indices;
                    }
                    std::span<const index_t> bias_neurons_indices() const {
                        return _net->_bias_neurons_indices;
                    }
                protected:
                    explicit View(NNet& net) : _net(&net) {}
                private:
                    friend class NNet;
                    NNet* _net{nullptr};
            };
            class State {
                public:
                    explicit State(const int_t neuron_count) : _signals(neuron_count, 0.0_real), _weighted_sums(neuron_count, 0.0_real) {}
                    std::span<real_t> signals() {
                        return _signals;
                    }
                    std::span<const real_t> signals() const {
                        return _signals;
                    }
                    std::span<real_t> weighted_sums() {
                        return _weighted_sums;
                    }
                    std::span<const real_t> weighted_sums() const {
                        return _weighted_sums;
                    }
                    void seed(const View &view, const DataType auto &input) {
                        const auto neurons = view.neurons();
                        if ((input.size() != view.input_neurons_indices().size()) or 
                            (_signals.size() != neurons.size()) or 
                            (_weighted_sums.size() != neurons.size())) {
                            throw std::invalid_argument("NNet::State::seed: size error!");
                        }
                        for (const auto &[input_neuron_index, input_value] : std::views::zip(view.input_neurons_indices(), input)) {
                            _weighted_sums[input_neuron_index] = input_value;
                            _signals[input_neuron_index] = neurons[input_neuron_index].activate(input_value);
                        }
                        for (const auto &bias_neuron_index : view.bias_neurons_indices()) {
                            _weighted_sums[bias_neuron_index] = 1.0_real;
                            _signals[bias_neuron_index] = neurons[bias_neuron_index].activate(1.0_real);
                        }
                    }
                private:
                    std::vector<real_t> _signals{};
                    std::vector<real_t> _weighted_sums{};
            };
        public:
            // Constructor
            NNet(Topology topology) : _neurons(std::move(topology.neurons)), _connections(std::move(topology.connections)), _weights(_connections.size(), 0.0_real) {
                init();
            }
            // Destructor
            virtual ~NNet() = default;
            // Create a view of the net
            View view() {
                return View(*this);
            }
            // Train net using a set of training samples
            template <typename Self, TrainingDataType Data, LearningRuleType<typename Self::View, Data> LearningRule>
                void train(this Self &self, const Data &data, LearningRule rule) {
                    if (self._trained) {
                        std::println("NNet::train: net is already trained!");
                        return;
                    }
                    constexpr int_t max_epochs{1'000'000};
                    int_t epoch{0};
                    utils::Timer timer{};
                    auto view = self.view();
                    std::println("NNet::train: ==================================");
                    std::println("NNet::train: Training net with {} samples...   ", std::ranges::size(data.inputs));
                    std::println("NNet::train: ==================================");
                    timer.start();
                    self._weights = utils::random::generate<real_t>(std::ssize(self._weights), -0.1_real, +0.1_real);
                    rule.init(view, data);
                    while (not self._trained and (epoch <= max_epochs)) {
                        self._trained = rule.learn();
                        epoch++;
                        std::println("NNet::train: epoch: {}, elapsed time: {:.5f}s", epoch, timer.get_elapsed_time_s());
                    }
                    rule.apply(self._weights);
                    if (self._trained) {
                        std::println("NNet::train: Training summary:");
                        std::println("NNet::train: elapsed time: {:.5f}s", timer.get_elapsed_time_s());
                        std::println("NNet::train: epochs: {}", epoch);
                    }
                    else {
                        std::println("NNet::train: net training failed!");
                    }
                }
            // Infer from data
            template <typename Self, DatasetType Inputs, net::update::BackendType<Inputs, dataset_vector_t, typename Self::View> Backend>
                dataset_vector_t infer(this Self &self, const Inputs &inputs, Backend backend) {
                    if (not self._trained) {
                        std::println("NNet::infer: try to infer from an untrained net!");
                        return {};
                    }
                    auto view = self.view();
                    dataset_vector_t outputs;
                    backend.init(view);
                    backend.to_device(inputs);
                    backend.update_states(view.weights());
                    backend.to_host(outputs);
                    return outputs;
                }
        protected: 
            // Data types
            using IndexPair = std::pair<index_t, index_t>;
            struct IndexPairHash {
                int_t operator()(const IndexPair &pair) const {
                    return std::hash<index_t>{}(pair.first) ^ (std::hash<index_t>{}(pair.second) << 1);
                }
            };
            // Getters
            std::span<const Neuron> neurons() const {
                return _neurons;
            }
            std::span<const Partition> partitions() const {
                return _partitions;
            }
            std::span<const Connection> connections() const {
                return _connections;
            }
            std::span<const real_t> weights() const {
                return _weights;
            }
            std::span<Partition> partitions() {
                return _partitions;
            }
        private:
            // Initialize net
            void init() {
                const auto neuron_count = std::ssize(_neurons);
                for (const auto &[tx_index, rx_index] : _connections) {
                    if ((tx_index < 0) or (tx_index >= neuron_count) or (rx_index < 0) or (rx_index >= neuron_count)) {
                        throw std::out_of_range("NNet::init: connection index out-of-range!");
                    }
                }
                if (has_duplicate_connection(_connections)) {
                    throw std::invalid_argument("NNet::init: duplicate connection!");
                }
                for (const auto &[neuron_index, neuron] : _neurons | std::views::enumerate) {
                    switch (neuron.type()) {
                        case NeuronType::input:
                            _input_neurons_indices.push_back(neuron_index);
                            break;
                        case NeuronType::output:
                            _output_neurons_indices.push_back(neuron_index);
                            break;
                        case NeuronType::bias:
                            _bias_neurons_indices.push_back(neuron_index);
                            break;
                        default:
                            break;
                    }
                }
                _partitions = make_partitions(_connections);
                std::println("NNet::init: neuron(s): {}, connection(s): {}, partition(s): {}", _neurons.size(), _connections.size(), _partitions.size());
            }
            // Check for duplicate connections
            static bool has_duplicate_connection(const std::span<Connection> connections) {
                std::unordered_map<IndexPair, index_t, IndexPairHash> hash_map;
                for (const auto &[connection_index, connection] : connections | std::views::enumerate) {
                    const auto [_, inserted] = hash_map.try_emplace(std::pair{connection.rx_index, connection.tx_index}, connection_index);
                    if (not inserted) {
                        return true;
                    }
                }
                return false;
            }
            // Group connections sharing the same rx into partitions
            static std::vector<Partition> make_partitions(const std::span<Connection> connections) {
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
            // Data members
            bool _trained{false};
            std::vector<Neuron> _neurons{};
            std::vector<Connection> _connections{};
            std::vector<real_t> _weights{};
            std::vector<Partition> _partitions{};
            std::vector<index_t> _input_neurons_indices{};
            std::vector<index_t> _output_neurons_indices{};
            std::vector<index_t> _bias_neurons_indices{};
    };
}
