#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning-rule.h"
#include "hnnet/builtin/activations.h"
#include "timer.h"
#include <numeric>
#include <omp.h>

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
                struct Partition {
                    index_t irx;
                    size_t begin;
                    size_t end;
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
                            return _net._itxs.size();
                        }
                        const Neuron& neuron(const index_t index) const {
                            return _net._neurons[index];
                        }
                        // Transmitting/receiving neuron index and weight for a given connection
                        index_t itx(const index_t iconn) const {
                            return _net._itxs[iconn];
                        }
                        index_t irx(const index_t iconn) const {
                            return _net._irxs[iconn];
                        }
                        real_t& weight(const index_t iconn) {
                            return _net._weights[iconn];
                        }
                        index_t neuron_index(const Neuron* neuron) const {
                            return _net.neuron_index(neuron);
                        }
                        const std::vector<Partition>& partitions() const {
                            return _net._partitions;
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
                    _itxs.reserve(_neurons.capacity() * 10);
                    _irxs.reserve(_neurons.capacity() * 10);
                    _weights.reserve(_neurons.capacity() * 10);
                }
                // Create a view of the net
                View view() {
                    return View(*this);
                }
                // Create new neurons
                template <ActivationType Activation>
                    auto new_neurons(const size_t number, const NeuronType &type, const Activation &activation) {
                        _trained = false;
                        for (size_t i{0}; i < number; ++i) {
                            _neurons.push_back({type, std::make_unique<Activation>(activation)});
                        }
                        return _neurons | std::views::drop(_neurons.size() - number) | std::views::take(number);
                }
                // Connect neurons (cartesian product)
                template <typename TxType, typename RxType>
                    requires (NeuronView<std::remove_cvref_t<TxType>> or std::same_as<std::remove_cvref_t<TxType>, Neuron>) and
                             (NeuronView<std::remove_cvref_t<RxType>> or std::same_as<std::remove_cvref_t<RxType>, Neuron>)
                    void connect(TxType &&tx_arg, RxType &&rx_arg) {
                        _trained = false;
                        auto to_neuron_index = [this] (auto &&arg) {
                            if constexpr (std::same_as<std::remove_cvref_t<decltype(arg)>, Neuron>) {
                                return std::views::single(neuron_index(&arg));
                            }
                            else {
                                return std::forward<decltype(arg)>(arg) | std::views::transform([this] (const Neuron &neuron) { return neuron_index(&neuron); });
                            }
                        };
                        auto itxs = to_neuron_index(std::forward<TxType>(tx_arg));
                        auto irxs = to_neuron_index(std::forward<RxType>(rx_arg));
                        for (const auto &[itx, irx] : std::views::cartesian_product(itxs, irxs)) {
                            // review... 
                            const auto found = std::ranges::any_of(std::views::iota(index_t{0}, static_cast<index_t>(_itxs.size())),
                                                                   [&] (const auto iconn) { return _itxs[iconn] == itx and _irxs[iconn] == irx; });
                            if (found) {
                                throw std::invalid_argument("NNet::connect: duplicate connection!");
                            }
                            _itxs.push_back(itx);
                            _irxs.push_back(irx);
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
                // Add bias to neurons
                template <NeuronView View>
                    void add_bias(View neurons) {
                        auto bias = new_neurons(1, NeuronType::bias, Builtin::IdentityActivation{});
                        connect(bias, neurons);
                        bias.front().activate(1.0);
                    }
                // Set weights to random values in the range [min, max]
                void randomize_weights(const real_t min, const real_t max) {
                    std::uniform_real_distribution<real_t> dist(min, max);
                    for (auto &weight : _weights) {
                        weight = dist(random_generator());
                    }
                }
                // Train net using a set of training samples
                template <typename LearningRule>
                    requires LearningRuleType<LearningRule, NNet>
                    void train(const std::vector<TrainingSample> &samples, LearningRule rule) {
                        constexpr real_t error_threshold{1e-2};
                        constexpr size_t max_epochs{1'000'000};
                        size_t epoch{0};
                        bool converged{false};
                        Timer timer{};
                        auto in_neurons = neurons(NeuronType::input);
                        auto out_neurons = neurons(NeuronType::output);
                        if (std::ranges::distance(in_neurons) != input_size or std::ranges::distance(out_neurons) != output_size) {
                            throw std::invalid_argument("NNet::train: size error!");
                        }
                        std::println("NNet::train: ==================================");
                        std::println("NNet::train: Training net with {} samples...   ", samples.size());
                        std::println("NNet::train: ==================================");
                        timer.start();
                        prepare();
                        while (not converged and (epoch <= max_epochs)) {
                            Timer epoch_timer{};
                            epoch_timer.start();
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
                            //std::ranges::shuffle(samples, random_generator()); -- samples must be not const!
                            for (const auto &sample : samples) {
                                reset();
                                inject(sample.inputs, in_neurons);
                                broadcast();
                                learn(sample.targets, rule);
                                for (const auto &[neuron, target] : std::views::zip(out_neurons, sample.targets)) { // delegate err computation to learning rule??
                                    const auto error = (target - neuron.signal());
                                    mean_squared_err += error * error;
                                }
                            }
                            epoch_timer.stop();
                            mean_squared_err /= samples.size();
                            converged = (mean_squared_err < error_threshold);
                            std::println("NNet::train: epoch: {}, elapsed time: {}s, mean squared error: {:.6f}", epoch, epoch_timer.get_elapsed_time_s(), mean_squared_err);
                        }
                        timer.stop();
                        if (converged) {
                            _trained = true;
                            std::println("NNet::train: Training summary:");
                            //for (const auto &[i, weight] : _weights | std::views::enumerate) {
                            //    std::println("NNet::train: weight[{}]: {}", i, weight);
                            //}
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
                    auto in_neurons = neurons(NeuronType::input);
                    auto out_neurons = neurons(NeuronType::output);
                    if (std::ranges::distance(in_neurons) != input_size or std::ranges::distance(out_neurons) != output_size) {
                        throw std::invalid_argument("NNet::infer: size error!");
                    }
                    reset();
                    inject(data, in_neurons);
                    broadcast();
                    OutputData outputs;
                    for (const auto &[idx, neuron] : out_neurons | std::views::enumerate) {
                        outputs[idx] = neuron.signal(); 
                    }
                    return outputs;
                }
            private:
                // Inject input data into neurons
                template <NeuronView View>
                    void inject(const InputData &inputs, View neurons) {
                        if (std::ranges::distance(neurons) != std::ranges::distance(inputs)) {
                            throw std::invalid_argument("NNet::inject: size error!");
                        }
                        for (const auto &[neuron, input] : std::views::zip(neurons, inputs)) {
                            neuron.activate(input);
                        }
                    }
                // Broadcast signals through the net using the partitions, already in topological order
                void broadcast() {
                    for (const auto &partition : _partitions) {
                        broadcast(partition);
                    }
                }
                // Process a single partition
                void broadcast(const Partition &partition) {
                    real_t weighted_sum{0.0};
                    size_t iconn{partition.begin};
                    for (; (iconn + register_size) <= partition.end; iconn += register_size) {
                        weighted_sum +=  _weights[iconn]     * _neurons[_itxs[iconn]].signal()
                                       + _weights[iconn + 1] * _neurons[_itxs[iconn + 1]].signal()
                                       + _weights[iconn + 2] * _neurons[_itxs[iconn + 2]].signal()
                                       + _weights[iconn + 3] * _neurons[_itxs[iconn + 3]].signal();
                    }
                    for (; iconn < partition.end; iconn++) {
                        weighted_sum += _weights[iconn] * _neurons[_itxs[iconn]].signal();
                    }
                    auto &receiver = _neurons[partition.irx];
                    receiver.activate(weighted_sum);
                }
                // Sort connections by (irx, itx), build per-receiver partitions and order them topologically
                void prepare() {
                    const auto connection_count = _itxs.size();
                    std::vector<index_t> order(connection_count);
                    std::vector<index_t> sorted_itxs(connection_count);
                    std::vector<index_t> sorted_irxs(connection_count);
                    std::vector<real_t> sorted_weights(connection_count);
                    Timer timer{};
                    std::println("NNet::prepare: preparing net...");
                    timer.start();
                    std::iota(order.begin(), order.end(), 0);
                    std::ranges::sort(order, [&] (const auto &lhs, const auto &rhs) { return std::tie(_irxs[lhs], _itxs[lhs]) < std::tie(_irxs[rhs], _itxs[rhs]); });
                    for (size_t i{0}; i < connection_count; ++i) {
                        const auto old_index = order[i];
                        sorted_itxs[i] = _itxs[old_index];
                        sorted_irxs[i] = _irxs[old_index];
                        sorted_weights[i] = _weights[old_index];
                    }
                    _itxs = std::move(sorted_itxs);
                    _irxs = std::move(sorted_irxs);
                    _weights = std::move(sorted_weights);
                    // Each contiguous block sharing the same irx becomes a partition
                    _partitions.clear();
                    size_t begin{0};
                    while (begin < _irxs.size()) {
                        const auto irx = _irxs[begin];
                        size_t end{begin + 1};
                        while (end < _irxs.size() and _irxs[end] == irx) {
                            ++end;
                        }
                        _partitions.push_back({.irx = irx, .begin = begin, .end = end});
                        begin = end;
                    }
                    // Topologically order partitions (Kahn's algorithm) so a single sequential pass suffices
                    const auto neuron_count = _neurons.size();
                    std::vector<size_t> in_degree(neuron_count, 0);
                    for (const auto &partition : _partitions) {
                        in_degree[partition.irx] = partition.end - partition.begin;
                    }
                    std::vector<std::vector<index_t>> out_adjacency(neuron_count);
                    for (index_t iconn{0}; iconn < static_cast<index_t>(_itxs.size()); ++iconn) {
                        out_adjacency[_itxs[iconn]].push_back(_irxs[iconn]);
                    }
                    std::vector<size_t> topo_rank(neuron_count, 0);
                    std::vector<index_t> ready;
                    for (index_t neuron{0}; neuron < static_cast<index_t>(neuron_count); ++neuron) {
                        if (in_degree[neuron] == 0) {
                            ready.push_back(neuron);
                        }
                    }
                    size_t processed_count{0};
                    while (not ready.empty()) {
                        const auto neuron = ready.back();
                        ready.pop_back();
                        topo_rank[neuron] = processed_count++;
                        for (const auto downstream : out_adjacency[neuron]) {
                            if (--in_degree[downstream] == 0) {
                                ready.push_back(downstream);
                            }
                        }
                    }
                    if (processed_count != neuron_count) {
                        throw std::runtime_error("NNet::prepare: net contains a cycle, topological order does not exist!");
                    }
                    std::ranges::sort(_partitions, [&] (const auto &lhs, const auto &rhs) { return topo_rank[lhs.irx] < topo_rank[rhs.irx]; });
                    //for (const auto partition : _partitions) {                                                                                 
                    //    std::println("irx: {}, begin: {}, end: {}", partition.irx, partition.begin, partition.end);                            
                    //    for (const auto &iconn : std::views::iota(partition.begin, partition.end)) {                                           
                    //        std::println("irx: {}, itx: {}", _irxs[iconn], _itxs[iconn]);                                                      
                    //    }                                                                                                                      
                    //}    
                    timer.stop();
                    std::println("NNet::prepare: net prepared in {}s.", timer.get_elapsed_time_s());
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
                        if (neuron.type() != NeuronType::bias) {
                            neuron.reset();
                        }
                    }
                }
                // Get neurons of given type
                auto neurons(const NeuronType type) {
                    return _neurons | std::views::filter([type] (const auto &neuron) { return neuron.type() == type; });
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
                std::vector<index_t> _itxs{};
                std::vector<index_t> _irxs{};
                std::vector<real_t> _weights{};
                std::vector<Partition> _partitions{};

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
