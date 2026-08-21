#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning-rule.h"
#include "timer.h"

// TODO: 1) Define a SourceNeuron class for input neurons that can receive external data and broadcast it to the network.
//          Add the requirement input_neurons to be SourceNeurons in NNet::feed() and NNet::train().

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
            // Data types
            struct TrainingSample {
                InputData inputs;
                OutputData targets;
            };
            struct SynapticConn {
                Neuron* const tx{nullptr};
                Neuron* const rx{nullptr};
                real_t weight{0.0};
            };
            ////////////////
            // View class //
            ////////////////
            class View {
                public:
                    size_t neuron_count() const {
                        return _net._neurons.size();
                    }
                    size_t connection_count() const {
                        return _net._connections.size();
                    }
                    const Neuron& neuron(const index_t index) const {
                        return _net._neurons[index];
                    }
                    const auto& in_connections(const index_t index) const {
                        return _net._in_connections[index];
                    }
                    const auto& out_connections(const index_t index) const {
                        return _net._out_connections[index];
                    }
                    SynapticConn& connection(const index_t index) {
                        return _net._connections[index];
                    }
                    index_t neuron_index(const Neuron* neuron) const {
                        return _net.neuron_index(neuron);
                    }
                    // Get indices of input neurons
                    auto in_neurons_indices() const {
                        return _net.input_neurons() | std::views::transform([this] (const auto &neuron) { return neuron_index(&neuron); });
                    }
                    // Get indices of output neurons
                    auto out_neurons_indices() const {
                        return _net.output_neurons() | std::views::transform([this] (const auto &neuron) { return neuron_index(&neuron); });
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
                _connections.reserve(_neurons.capacity() * 10);
            }
            // Create a view of the net
            View view() {
                return View(*this);
            }
            // Get neurons of given type
            auto neurons(const NeuronType type) {
                return _neurons | std::views::filter([type] (const auto &neuron) { return neuron.type() == type; });
            }
            // Get input neurons
            auto input_neurons() {
                auto in_neurons = neurons(NeuronType::input);
                if (std::ranges::distance(in_neurons) != input_size) {
                    throw std::runtime_error("NNet::input_neurons: size error!");
                }
                return in_neurons;
            }
            // Get output neurons
            auto output_neurons() { 
                auto out_neurons = neurons(NeuronType::output);
                if (std::ranges::distance(out_neurons) != output_size) {
                    throw std::runtime_error("NNet::output_neurons: size error!");
                }
                return out_neurons;
            }
            // Create new neurons (all constructed with the same arguments)
            template <typename... Args>
                requires (std::constructible_from<Neuron, Args...>)
                auto new_neurons(const size_t number, Args &&...args) {
                    for (size_t i{0}; i < number; ++i) {
                        _neurons.emplace_back(std::forward<Args>(args)...);
                        _in_connections.emplace_back();
                        _out_connections.emplace_back();
                    }
                    return _neurons | std::views::drop(_neurons.size() - number) | std::views::take(number);
                }
            // Connect neurons: each side can be a single Neuron or a NeuronRange
            template <typename Tx, typename Rx>
                requires (NeuronRange<Tx> or std::same_as<std::remove_cvref_t<Tx>, Neuron>) and
                         (NeuronRange<Rx> or std::same_as<std::remove_cvref_t<Rx>, Neuron>)
                void connect(Tx &&tx_arg, Rx &&rx_arg) {
                    auto to_neuron_ptrs = [] (auto &&arg) {
                        if constexpr (std::same_as<std::remove_cvref_t<decltype(arg)>, Neuron>) {
                            return std::views::single(&arg);
                        } else {
                            return std::forward<decltype(arg)>(arg) | std::views::transform([] (Neuron &neuron) { return &neuron; });
                        }
                    };
                    auto tx_neurons = to_neuron_ptrs(std::forward<Tx>(tx_arg));
                    auto rx_neurons = to_neuron_ptrs(std::forward<Rx>(rx_arg));
                    auto in_net = [&] (const Neuron *neuron) {
                        return std::ranges::any_of(_neurons, [&] (const auto &net_neuron) { return (&net_neuron == neuron); });
                    };
                    if (not std::ranges::all_of(tx_neurons, in_net) or not std::ranges::all_of(rx_neurons, in_net)) {
                        throw std::invalid_argument("NNet::connect: neurons not in net!");
                    }
                    for (const auto &[tx, rx] : std::views::cartesian_product(tx_neurons, rx_neurons)) {
                        const auto found = std::ranges::any_of(_connections, [&] (const auto &conn) { return (conn.tx == tx) and (conn.rx == rx); });
                        if (found) {
                            throw std::invalid_argument("NNet::connect: duplicate connection!");
                        }
                        _connections.emplace_back(tx, rx, 0.0);
                        const auto iconn = _connections.size() - 1;
                        _out_connections[neuron_index(tx)].emplace_back(iconn);
                        _in_connections[neuron_index(rx)].emplace_back(iconn);
                    }
                }
            // Add bias to neurons
            template <NeuronView View>
                void add_bias(View neurons) {
                    auto bias_neurons = new_neurons(std::ranges::distance(neurons), NeuronType::bias);
                    for (const auto &[bias, neuron] : std::views::zip(bias_neurons, neurons)) {
                        connect(bias, neuron);
                        bias.receive_signal(1.0);
                        bias.activate();
                    }
                }
            // Set weights to random values in the range [min, max]
            void randomize_weights(const real_t min, const real_t max) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<real_t> dist(min, max);
                for (const auto &[idx, conn] : _connections | std::views::enumerate) {
                    conn.weight = dist(gen);
                    std::println("NNet::randomize_weights: weight[{}]: {}", idx, conn.weight);
                }
            }    
            // Train net using a set of training samples
            template <typename LearningRule>
                requires LearningRuleType<LearningRule, NNet>
                void train(const std::vector<TrainingSample> &samples, LearningRule rule) {
                    constexpr real_t error_threshold{5e-2};
                    constexpr size_t max_epochs{1'000'000};
                    auto in_neurons = input_neurons();
                    auto out_neurons = output_neurons();
                    auto bias_neurons = neurons(NeuronType::bias);
                    size_t epoch{0};
                    bool converged{false};
                    Timer timer{};
                    std::println("NNet::train: ==================================");
                    std::println("NNet::train: Training net with {} samples...   ", samples.size());
                    std::println("NNet::train: ==================================");
                    timer.start();
                    while (not converged and (epoch <= max_epochs)) {
                        epoch++;
                        if ((epoch < 10        and epoch % 1 == 0)      or
                            (epoch < 100       and epoch % 10 == 0)     or
                            (epoch < 1000      and epoch % 100 == 0)    or 
                            (epoch < 10'000    and epoch % 1000 == 0)   or
                            (epoch < 100'000   and epoch % 10'000 == 0) or
                            (epoch < 1'000'000 and epoch % 100'000 == 0)) {
                            std::println("NNet::train: Epoch: {}", epoch);
                        }
                        real_t mean_squared_err{0.0};
                        for (const auto &sample : samples) {
                            reset();
                            inject(sample.inputs, in_neurons);
                            broadcast(bias_neurons);
                            broadcast(in_neurons);
                            learn(sample.targets, rule);
                            for (const auto &[neuron, target] : std::views::zip(out_neurons, sample.targets)) {
                                const auto error = (target - neuron.signal());
                                mean_squared_err += error * error;
                            }
                        }
                        //mean_squared_err /= samples.size();
                        converged = (mean_squared_err < error_threshold);
                        //std::println("NNet::train: epoch: {}, mean squared error: {:.6f}", epoch, mean_squared_err);
                    }
                    timer.stop();
                    if (converged) {
                        _trained = true;
                        std::println("NNet::train: Training summary:");
                        for (const auto &[idx, conn] : _connections | std::views::enumerate) {
                            std::println("NNet::train: weight[{}]: {}", idx, conn.weight);
                        }
                        auto targets = samples | std::views::transform([&] (const auto &sample) { return sample.targets; });
                        auto outputs = samples | std::views::transform([&] (const auto &sample) { return infer(sample.inputs); });
                        std::println("NNet::train: targets: {}, outputs: {}", targets, outputs);
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
                auto in_neurons = input_neurons();
                auto bias_neurons = neurons(NeuronType::bias);
                reset();
                inject(data, in_neurons);
                broadcast(bias_neurons);
                broadcast(in_neurons);
                OutputData outputs;
                for (const auto &[idx, neuron] : output_neurons() | std::views::enumerate) {
                    outputs[idx] = neuron.signal(); 
                }
                return outputs;
            }
        private:
            // Inject input data into neurons
            template <NeuronView View>
                void inject(const InputData &inputs, View neurons) {
                    if (std::ranges::distance(neurons) != input_size) {
                        throw std::invalid_argument("NNet::inject: size error!");
                    }
                    for (const auto &[neuron, input] : std::views::zip(neurons, inputs)) {
                        neuron.receive_signal(input);
                        neuron.activate();
                    }
                }
            // Broadcast neuron signals through the network
            template <NeuronView View>
                void broadcast(View neurons) {
                    for (auto &neuron : neurons) {
                        broadcast(neuron);
                    }
                }
            void broadcast(const Neuron& neuron) {
                for (const auto &iconn : _out_connections[neuron_index(&neuron)]) {
                    auto& conn = _connections[iconn];
                    conn.rx->receive_signal(neuron.signal() * conn.weight);
                    if (conn.rx->number_rx_signals() == _in_connections[neuron_index(conn.rx)].size()) {
                        conn.rx->activate();
                        broadcast(*conn.rx);
                    }
                }
            }
            // Learn from targets using the selected learning rule
            template <typename LearningRule>
                requires LearningRuleType<LearningRule, NNet>
                void learn(const OutputData &targets, LearningRule &rule) {
                    rule.learn(*this, targets);
                }
            // Get the index of a neuron
            index_t neuron_index(const Neuron* neuron) const {
                const auto index = static_cast<index_t>(neuron - _neurons.data());
                if (index >= _neurons.size()) {
                    throw std::invalid_argument("NNet::neuron_index: invalid index!");
                }
                return index;
            }
            // Reset all neurons in the net
            void reset() {
                for (auto& neuron : _neurons) {
                    neuron.reset();
                }
            }
            // Data members
            bool _trained{false};
            std::vector<Neuron> _neurons{};
            std::vector<SynapticConn> _connections{};
            std::vector<std::vector<index_t>> _in_connections{};
            std::vector<std::vector<index_t>> _out_connections{};
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