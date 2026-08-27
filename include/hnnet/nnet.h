#pragma once
#include "hnnet/neuron.h"
#include "hnnet/learning-rule.h"
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
            // Create new neurons (all constructed with the same arguments)
            template <typename... Args>
                requires (std::constructible_from<Neuron, Args...>)
                auto new_neurons(const size_t number, Args &&...args) {
                    for (size_t i{0}; i < number; ++i) {
                        _neurons.push_back({std::forward<Args>(args)...});
                    }
                    return _neurons | std::views::drop(_neurons.size() - number) | std::views::take(number);
                }
            // Connect neurons (cartesian product)
            template <typename Tx, typename Rx>
                requires (NeuronView<std::remove_cvref_t<Tx>> or std::same_as<std::remove_cvref_t<Tx>, Neuron>) and
                         (NeuronView<std::remove_cvref_t<Rx>> or std::same_as<std::remove_cvref_t<Rx>, Neuron>)
                void connect(Tx &&tx_arg, Rx &&rx_arg) {
                    auto to_neuron_index = [this] (auto &&arg) {
                        if constexpr (std::same_as<std::remove_cvref_t<decltype(arg)>, Neuron>) {
                            return std::views::single(neuron_index(&arg));
                        } else {
                            return std::forward<decltype(arg)>(arg) | std::views::transform([this] (const Neuron &neuron) { return neuron_index(&neuron); });
                        }
                    };
                    auto itxs = to_neuron_index(std::forward<Tx>(tx_arg));
                    auto irxs = to_neuron_index(std::forward<Rx>(rx_arg));
                    for (const auto &[itx, irx] : std::views::cartesian_product(itxs, irxs)) {
                        //const auto found = std::ranges::any_of(std::views::iota(index_t{0}, static_cast<index_t>(_itxs.size())), [&] (const auto iconn) { return _itxs[iconn] == itx and _irxs[iconn] == irx; });
                        //if (found) {
                        //    throw std::invalid_argument("NNet::connect: duplicate connection!");
                        //}
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
                    auto bias = new_neurons(1, NeuronType::bias);
                    connect(bias, neurons);
                    bias.front().set_weighted_sum(1.0);
                    bias.front().activate();
                }
            // Sort connections by (irx, itx) and build per-receiver partitions; call after all connect()/add_bias()
            void prepare() {
                std::println("NNet::prepare: preparing net...");
                Timer timer{};
                timer.start();
                const auto connection_count = _itxs.size();
                std::vector<index_t> order(connection_count);
                std::iota(order.begin(), order.end(), 0);
                std::ranges::sort(order, [this] (const index_t lhs, const index_t rhs) { return std::tie(_irxs[lhs], _itxs[lhs]) < std::tie(_irxs[rhs], _itxs[rhs]); });
                std::vector<index_t> sorted_itxs(connection_count);
                std::vector<index_t> sorted_irxs(connection_count);
                std::vector<real_t> sorted_weights(connection_count);
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
                    while (end < _irxs.size() and _irxs[end] == irx) ++end;
                    _partitions.push_back({.irx = irx, .begin = begin, .end = end});
                    begin = end;
                }
                timer.stop();
                std::println("NNet::prepare: net prepared in {}s.", timer.get_elapsed_time_s());
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
                    std::println("NNet::train: ==================================");
                    std::println("NNet::train: Training net with {} samples...   ", samples.size());
                    std::println("NNet::train: ==================================");
                    Timer timer{};
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
                        //for (const auto &[idx, conn] : _connections | std::views::enumerate) {
                        //    std::println("NNet::train: weight[{}]: {}", idx, conn.weight);
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
            // A partition is a contiguous block of connections sharing the same irx (a sparse matrix row)
            struct Partition {
                index_t irx;
                size_t begin;
                size_t end;
            };
            // Inject input data into neurons
            template <NeuronView View>
                void inject(const InputData &inputs, View neurons) {
                    if (std::ranges::distance(neurons) != std::ranges::distance(inputs)) {
                        throw std::invalid_argument("NNet::inject: size error!");
                    }
                    for (const auto &[neuron, input] : std::views::zip(neurons, inputs)) {
                        neuron.set_weighted_sum(input);
                        neuron.activate();
                    }
                }
            // Broadcast signals through the network using the partitions built by prepare()
            void broadcast() {
                _partition_processed.assign(_partitions.size(), false);
                broadcast_partitions();
            }
            // Repeatedly scan not-yet-processed partitions until no more progress is made
            void broadcast_partitions() {
                std::vector<size_t> ready_partitions;
                for (const auto &[i, partition] : _partitions | std::views::enumerate) {
                    if (not _partition_processed[i] and std::ranges::all_of(std::views::iota(partition.begin, partition.end), [this] (const size_t iconn) { return _neurons[_itxs[iconn]].activated(); })) ready_partitions.push_back(i);
                }
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (size_t i = 0; i < ready_partitions.size(); ++i) {
                    broadcast(_partitions[ready_partitions[i]]);
                }
                bool progress{false};
                for (const auto i : ready_partitions) {
                    _partition_processed[i] = true;
                    progress = true;
                }
                if (progress) {
                    broadcast_partitions();
                    return;
                }
                const bool all_processed = std::ranges::all_of(_partition_processed, [] (const bool processed) { return processed; });
                if (not all_processed) {
                    throw std::runtime_error("NNet::broadcast: unable to complete propagation (cycle or unreachable neuron)!");
                }
            }
            // Process a single partition (a receiver's row); returns true if it was ready and got activated
            bool broadcast(const Partition &partition) {
                const bool ready = std::ranges::all_of(std::views::iota(partition.begin, partition.end), [this] (const size_t iconn) { return _neurons[_itxs[iconn]].activated(); });
                if (not ready) {
                    return false;
                }
                real_t weighted_sum{0.0};
                for (size_t iconn = partition.begin; iconn < partition.end; ++iconn) {
                    weighted_sum += _weights[iconn] * _neurons[_itxs[iconn]].signal();
                }
                auto &receiver = _neurons[partition.irx];
                receiver.set_weighted_sum(weighted_sum);
                receiver.activate();
                return true;
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
            std::vector<bool> _partition_processed{};
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
