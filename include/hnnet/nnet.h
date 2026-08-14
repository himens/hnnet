#pragma once
#include "hnnet/neuron.h"
#include "timer.h"

// TODO: 1) Define a SourceNeuron class for input neurons that can receive external data and broadcast it to the network.
//          Add the requirement in_neurons to be SourceNeurons in NNet::feed() and NNet::train().

namespace hNNet {
    ////////////////
    // NNet class //
    ////////////////
    template <DataType InputType, DataType OutputType>
    class NNet {
        public:
            // Data types 
            using InputData = InputType;   
            using OutputData = OutputType;
            struct TrainingData {
                InputData inputs;
                OutputData targets;
            };
            // Create a single new neuron
            template <NeuronType T>
                Neuron* new_neuron() {
                    auto neuron = std::make_unique<T>();
                    auto ptr = neuron.get();
                    _neurons.push_back(std::move(neuron));
                    return ptr;
                }            
            // Create new neurons
            template <NeuronType T>
                std::vector<Neuron*> new_neurons(const size_t number) {
                    if (number == 0) {
                        throw std::invalid_argument("NNet::new_neurons: number of neurons must be > 0!");
                    }
                    std::vector<Neuron*> neurons(number);
                    for (auto &neuron : neurons) {
                        neuron = new_neuron<T>();
                    }
                    return neurons;
                }
            // Connect neurons
            template <NeuronRange TxRange, NeuronRange RxRange>
                void connect(const TxRange &tx_neurons, const RxRange &rx_neurons) {
                    auto neurons = _neurons | std::views::transform([] (const auto &neuron) { return neuron.get(); });
                    const auto in_net = [&] (Neuron *neuron) {
                        return std::ranges::any_of(neurons, [&] (Neuron* net_neuron) { return net_neuron == neuron; });
                    };
                    if (not std::ranges::all_of(tx_neurons, in_net) or not std::ranges::all_of(rx_neurons, in_net)) {
                        throw std::invalid_argument("NNet::connect: neurons not in net!");
                    }
                    for (const auto &[tx, rx] : std::views::cartesian_product(tx_neurons, rx_neurons)) {
                        auto conn = std::make_unique<Neuron::SynapticConn>(tx, rx);
                        tx->add_connection(conn.get());
                        rx->add_connection(conn.get());
                        _connections.push_back(std::move(conn));
                    }
                }
            template <NeuronRange Range>
                void connect(const Range &tx_neurons, Neuron* rx) {
                    connect(tx_neurons, std::views::single(rx));
                }
            template <NeuronRange Range>
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
                    conn->weight = dist(gen);
                    std::println("NNet::randomize_weights: weight[{}]: {}", idx, conn->weight);
                }
            }    
            // Train net using a set of training samples
            void train(const std::vector<TrainingData> &training_samples) {
                constexpr real_t error_threshold{5e-2};
                constexpr size_t max_epochs{1'000'000};
                auto in_neurons  = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_in_connections().empty(); });
                auto out_neurons = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_out_connections().empty(); });
                if (std::ranges::distance(in_neurons)  != std::tuple_size<InputData>::value or
                    std::ranges::distance(out_neurons) != std::tuple_size<OutputData>::value) {
                    throw std::invalid_argument("NNet::train: size error!");
                }
                size_t epoch{0};
                bool converged{false};
                Timer timer{};
                std::println("NNet::train: ==================================");
                std::println("NNet::train: Training net with {} samples...   ", training_samples.size());
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
                        //std::println("NNet::train: Epoch: {}", epoch);
                    }
                    real_t mean_squared_err{0.0};
                    for (const auto &sample : training_samples) {
                        feed(in_neurons, sample.inputs);
                        for (const auto &[neuron, target] : std::views::zip(out_neurons, sample.targets)) {
                            const auto error = (target - neuron->get_signal());
                            neuron->learn(error);
                            mean_squared_err += error * error;
                        }
                    }
                    //mean_squared_err /= training_samples.size();
                    converged = (mean_squared_err < error_threshold);
                    //std::println("NNet::train: epoch: {}, mean squared error: {:.6f}", epoch, mean_squared_err);
                }
                timer.stop();
                if (converged) {
                    _trained = true;
                    std::println("NNet::train: Training summary:");
                    for (const auto &[idx, conn] : _connections | std::views::enumerate) {
                        std::println("NNet::train: weight[{}]: {}", idx, conn->weight);
                    }
                    auto targets = training_samples | std::views::transform([&] (const auto &sample) { return sample.targets; });
                    auto outputs = training_samples | std::views::transform([&] (const auto &sample) { return infer(sample.inputs); });
                    std::println("NNet::train: targets: {}, outputs: {}", targets, outputs);
                    std::println("NNet::train: elapsed time: {}s", timer.get_elapsed_time_s());
                    std::println("NNet::train: epochs: {}", epoch);
                }
                else {
                    _trained = false;
                    std::println("NNet::train: net training failed!");
                };
            }
            // Infer from data
            OutputData infer(const InputData &data) {
                if (not _trained) {
                    std::println("NNet::infer: try to infer from an untrained net!");
                    return {};
                }
                auto in_neurons  = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_in_connections().empty(); });
                auto out_neurons = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_out_connections().empty(); });
                feed(in_neurons, data);
                OutputData outputs;
                for (const auto &[idx, neuron] : out_neurons | std::views::enumerate) {
                    outputs[idx] = neuron->get_signal(); 
                }
                return outputs;
            }  
        private:
            // Feed net with input data
            template <NeuronView View>
                void feed(View in_neurons, const InputData &inputs) {
                    if (std::ranges::distance(in_neurons) != inputs.size()) {
                        throw std::invalid_argument("NNet::feed: size error!");
                    }
                    for (const auto &[neuron, input] : std::views::zip(in_neurons, inputs)) {
                        neuron->set_signal(input);
                        neuron->broadcast_signal();
                    }
                }
            // Data members
            bool _trained{false};
            std::vector<std::unique_ptr<Neuron>> _neurons{};
            std::vector<std::unique_ptr<Neuron::SynapticConn>> _connections{};
    };
}