#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning-rule.h"
#include "timer.h"

// TODO: 1) Define a SourceNeuron class for input neurons that can receive external data and broadcast it to the network.
//          Add the requirement in_neurons to be SourceNeurons in NNet::feed() and NNet::train().

namespace hNNet {
    ////////////////
    // NNet class //
    ////////////////
    template <NeuronType Neuron, DataType InputData, DataType OutputData>
    class NNet {
        public:
            // Aliases and constants
            using neuron_type = Neuron;
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
                private:
                    friend class NNet;
                    explicit View(NNet& net) : _net(net) {}
                    NNet& _net;
            };
        public:
            // Constructor
            NNet() {
                _neurons.reserve(1000);  // Preallocate space for 1000 neurons
                _connections.reserve(_neurons.capacity() * 10);  // Preallocate space for 10 connections each neuron    
            }
            // Create a view of the net
            View view() {
                return View(*this);
            }
            // Create a single new neuron
            template <typename... Args>
                requires (std::constructible_from<Neuron, Args...>)
                Neuron* new_neuron(Args &&...args) {
                    _neurons.emplace_back(std::forward<Args>(args)...);
                    _in_connections.emplace_back();
                    _out_connections.emplace_back();
                    return &_neurons.back();
                }
            // Create new neurons (all constructed with the same arguments)
            template <typename... Args>
                requires (std::constructible_from<Neuron, Args...>)
                std::vector<Neuron*> new_neurons(const size_t number, Args &&...args) {
                    std::vector<Neuron*> neurons(number);
                    for (auto &neuron : neurons) {
                        neuron = new_neuron(std::forward<Args>(args)...);
                    }
                    return neurons;
                }
            // Connect neurons
            template <NeuronPtrRange TxRange, NeuronPtrRange RxRange>
                void connect(const TxRange &tx_neurons, const RxRange &rx_neurons) {
                    auto in_net = [&] (const auto *neuron) { return std::ranges::any_of(_neurons, [&] (const auto &net_neuron) { return &net_neuron == neuron; }); };
                    if (not std::ranges::all_of(tx_neurons, in_net) or not std::ranges::all_of(rx_neurons, in_net)) {
                        throw std::invalid_argument("NNet::connect: neurons not in net!");
                    }
                    for (const auto &[tx, rx] : std::views::cartesian_product(tx_neurons, rx_neurons)) {
                        const auto found = std::ranges::any_of(_connections, [&] (const auto &conn) { return (conn.tx == tx) and (conn.rx == rx); });
                        if (found) {
                            throw std::invalid_argument("NNet::connect: duplicate connection!");
                        }
                        _connections.push_back({.tx = tx, .rx = rx, .weight = 0.0});
                        const auto iconn = _connections.size() - 1;
                        const auto itx = neuron_index(tx);
                        const auto irx = neuron_index(rx);
                        _out_connections[itx].push_back(iconn);
                        _in_connections[irx].push_back(iconn);
                    }
                }
            template <NeuronPtrRange Range>
                void connect(const Range &tx_neurons, Neuron* rx) {
                    connect(tx_neurons, std::views::single(rx));
                }
            template <NeuronPtrRange Range>
                void connect(Neuron* tx, const Range &rx_neurons) {
                    connect(std::views::single(tx), rx_neurons);
                }
            void connect(Neuron* tx, Neuron* rx) {
                connect(std::views::single(tx), std::views::single(rx));
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
                    auto in_neurons  = _neurons | std::views::filter([&] (auto &neuron) { return _in_connections[neuron_index(&neuron)].empty(); })  | std::views::all;
                    auto out_neurons = _neurons | std::views::filter([&] (auto &neuron) { return _out_connections[neuron_index(&neuron)].empty(); }) | std::views::all;
                    if ((std::ranges::distance(in_neurons) != input_size) or (std::ranges::distance(out_neurons) != output_size)) {
                        throw std::invalid_argument("NNet::train: size error!");
                    }
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
                auto in_neurons  = _neurons | std::views::filter([&] (auto &neuron) { return _in_connections[neuron_index(&neuron)].empty(); })  | std::views::all;
                auto out_neurons = _neurons | std::views::filter([&] (auto &neuron) { return _out_connections[neuron_index(&neuron)].empty(); }) | std::views::all;
                reset();
                inject(data, in_neurons);
                broadcast(in_neurons);
                OutputData outputs;
                for (const auto &[idx, neuron] : out_neurons | std::views::enumerate) {
                    outputs[idx] = neuron.signal(); 
                }
                return outputs;
            }
        private:
            // Get the index of a neuron in the _neurons vector
            index_t neuron_index(const Neuron* neuron) const {
                return static_cast<index_t>(neuron - _neurons.data());
            }
            // Reset all neurons in the network
            void reset() {
                for (auto& neuron : _neurons) {
                    neuron.reset();
                }
            }
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
            // Data members
            bool _trained{false};
            std::vector<Neuron> _neurons{};
            std::vector<SynapticConn> _connections{};
            std::vector<std::vector<index_t>> _in_connections{};
            std::vector<std::vector<index_t>> _out_connections{};
    };
    template <typename T>
        using neuron_t = T::neuron_type;
    template <typename T>
        using input_t = T::input_type;
    template <typename T>
        using output_t = T::output_type;
    template<typename T>
        constexpr std::size_t input_size_v = T::input_size;
    template<typename T>
        constexpr std::size_t output_size_v = T::output_size;
    template <typename T>
        concept NNetType = std::derived_from<T, NNet<neuron_t<T>, input_t<T>, output_t<T>>>;
}