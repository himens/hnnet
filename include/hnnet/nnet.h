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
                        index_t itx(const index_t icon) const {
                            return _net._itxs[icon];
                        }
                        index_t irx(const index_t icon) const {
                            return _net._irxs[icon];
                        }
                        real_t& weight(const index_t icon) {
                            return _net._weights[icon];
                        }
                        const std::vector<Partition>& partitions() const {
                            return _net._partitions;
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
                    void connect(TxType &&tx_neurons, RxType &&rx_neurons) {
                        auto index = [&] (const Neuron* neuron) {
                            const auto index = static_cast<index_t>(neuron - _neurons.data());
                            if (index >= _neurons.size()) {
                                throw std::invalid_argument("NNet::connect::index: invalid index!");
                            }
                            return index;
                        };
                        auto to_index = [&] (auto &&arg) {
                            if constexpr (std::same_as<std::remove_cvref_t<decltype(arg)>, Neuron>) {
                                return std::views::single(index(&arg));
                            }
                            else {
                                return std::forward<decltype(arg)>(arg) | std::views::transform([&] (const auto &neuron) { return index(&neuron); });
                            }
                        };
                        _trained = false;
                        auto itxs = to_index(std::forward<TxType>(tx_neurons));
                        auto irxs = to_index(std::forward<RxType>(rx_neurons));
                        for (const auto &[itx, irx] : std::views::cartesian_product(itxs, irxs)) {
                            const auto found = std::ranges::any_of(std::views::iota(0ul, _irxs.size()), [&] (const auto &icon)
                                                                   { return (_itxs[icon] == itx) and (_irxs[icon]) == irx; });
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
                        auto in_neurons = neurons(NeuronType::input);
                        auto out_neurons = neurons(NeuronType::output);
                        if (std::ranges::distance(in_neurons) != input_size or std::ranges::distance(out_neurons) != output_size) {
                            throw std::invalid_argument("NNet::train: size error!");
                        }
                        Timer timer{};
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
                            real_t mean_squared_error{0.0};
                            //std::ranges::shuffle(samples, random_generator()); -- samples must be not const!
                            for (const auto &sample : samples) {
                                reset();
                                inject(sample.inputs, in_neurons);
                                broadcast();
                                mean_squared_error += learn(sample.targets, rule);
                            }
                            epoch_timer.stop();
                            mean_squared_error /= samples.size();
                            converged = (mean_squared_error < error_threshold);
                            std::println("NNet::train: epoch: {}, elapsed time: {}s, mean squared error: {:.6f}", epoch, epoch_timer.get_elapsed_time_s(), mean_squared_error);
                        }
                        timer.stop();
                        if (converged) {
                            _trained = true;
                            std::println("NNet::train: Training summary:");
                            //for (const auto &[i, weight] : _weights | std::views::enumerate) {
                            //    std::println("NNet::train: weight[{}]: {}", i, weight);
                            //}
                            //auto targets = samples | std::views::transform([&] (const auto &sample) { return sample.targets; });
                            //auto outputs = samples | std::views::transform([&] (const auto &sample) { return infer(sample.inputs); });
                            //std::println("NNet::train: targets: {}, outputs: {}", targets, outputs);
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
                    for (const auto &[iout, out_neuron] : out_neurons | std::views::enumerate) {
                        outputs[iout] = out_neuron.signal(); 
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
                    size_t icon{partition.begin};
                    for (; (icon + register_size) <= partition.end; icon += register_size) {
                        weighted_sum +=  _weights[icon]     * _neurons[_itxs[icon]].signal()
                                       + _weights[icon + 1] * _neurons[_itxs[icon + 1]].signal()
                                       + _weights[icon + 2] * _neurons[_itxs[icon + 2]].signal()
                                       + _weights[icon + 3] * _neurons[_itxs[icon + 3]].signal();
                    }
                    for (; icon < partition.end; icon++) {
                        weighted_sum += _weights[icon] * _neurons[_itxs[icon]].signal();
                    }
                    auto &rx = _neurons[partition.irx];
                    rx.activate(weighted_sum);
                }
                // Prepare net (build partitions, order them topologically, ...)
                void prepare() {
                    Timer timer;
                    std::println("NNet::prepare: preparing net...");
                    timer.start();
                    // sort connections per irx and itx
                    const auto connection_count = _itxs.size();
                    std::vector<index_t> sorted_icons(connection_count);
                    std::ranges::iota(sorted_icons, 0);
                    std::ranges::sort(sorted_icons, [&] (const auto &lhs, const auto &rhs)
                                      { return std::tie(_irxs[lhs], _itxs[lhs]) < std::tie(_irxs[rhs], _itxs[rhs]); });
                    std::vector<index_t> sorted_itxs(connection_count);
                    std::vector<index_t> sorted_irxs(connection_count);
                    for (size_t icon{0}; icon < connection_count; ++icon) {
                        const auto &sorted_icon = sorted_icons[icon];
                        sorted_itxs[icon] = _itxs[sorted_icon];
                        sorted_irxs[icon] = _irxs[sorted_icon];
                    }
                    _itxs = std::move(sorted_itxs);
                    _irxs = std::move(sorted_irxs);
                    // each contiguous block sharing the same irx becomes a partition
                    _partitions.clear();
                    size_t begin{0};
                    while (begin < _irxs.size()) {
                        const auto &irx = _irxs[begin];
                        size_t end{begin + 1};
                        while (end < _irxs.size() and _irxs[end] == irx) {
                            ++end;
                        }
                        _partitions.push_back({.irx = irx, .begin = begin, .end = end});
                        begin = end;
                    }
                    // save indices of output neurons
                    const auto neuron_count = _neurons.size();
                    _iout_neurons.clear();
                    for (index_t inr{0}; inr < neuron_count; ++inr) {
                        if (_neurons[inr].type() == NeuronType::output) {
                            _iout_neurons.push_back(inr);
                        }
                    }
                    // topologically order partitions (Kahn's algorithm)
                    std::vector<std::vector<index_t>> irxs(neuron_count);
                    for (index_t icon{0}; icon < static_cast<index_t>(_itxs.size()); ++icon) {
                        irxs[_itxs[icon]].push_back(_irxs[icon]);
                    }
                    std::vector<size_t> visits_left(neuron_count, 0);
                    for (const auto &partition : _partitions) {
                        visits_left[partition.irx] = partition.end - partition.begin;
                    }
                    std::vector<index_t> visited_queue;
                    for (index_t inr{0}; inr < static_cast<index_t>(neuron_count); ++inr) {
                        if (visits_left[inr] == 0) {
                            visited_queue.push_back(inr);
                        }
                    }
                    size_t visited_count{0};
                    std::vector<size_t> topo_rank(neuron_count, 0);
                    while (not visited_queue.empty()) {
                        const auto &inr = visited_queue.back();
                        visited_queue.pop_back();
                        topo_rank[inr] = visited_count++;
                        for (const auto &irx : irxs[inr]) {
                            if (--visits_left[irx] == 0) {
                                visited_queue.push_back(irx);
                            }
                        }
                    }
                    if (visited_count != neuron_count) {
                        throw std::runtime_error("NNet::prepare: net contains a cycle, topological order does not exist!");
                    }
                    std::ranges::sort(_partitions, [&] (const auto &lhs, const auto &rhs) { return topo_rank[lhs.irx] < topo_rank[rhs.irx]; });
                    //for (const auto partition : _partitions) {                                                                                 
                    //    std::println("irx: {}, begin: {}, end: {}", partition.irx, partition.begin, partition.end);
                    //    for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                    //        std::println("irx: {}, itx: {}", _irxs[icon], _itxs[icon]);
                    //    }                                                                                                                      
                    //}    
                    timer.stop();
                    std::println("NNet::prepare: net prepared in {}s", timer.get_elapsed_time_s());
                }
                // Learn from targets using the selected learning rule
                template <typename LearningRule>
                    requires LearningRuleType<LearningRule, NNet>
                    real_t learn(const OutputData &targets, LearningRule &rule) {
                        return rule.learn(*this, targets);
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
                std::vector<index_t> _iout_neurons{};
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
