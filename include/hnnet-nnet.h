#pragma once
#include "hnnet-neuron.h"
#include "timer.h"

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
            // Train net using a set of training samples
            void train(const std::vector<TrainingData> &training_samples) {
                auto in_neurons  = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_in_connections().empty(); });
                auto out_neurons = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_out_connections().empty(); });
                constexpr size_t max_epochs{1000};
                size_t epoch{0};
                bool converged{false};
                std::println("NNet::train: ==================================");
                std::println("NNet::train: Training net with {} samples...    ", training_samples.size());
                std::println("NNet::train: ==================================");
                Timer timer{};
                timer.start();
                while (not converged and (epoch <= max_epochs)) {
                    epoch++;
                    if ((epoch < 10      and epoch % 1 == 0)    or
                        (epoch < 100     and epoch % 10 == 0)   or
                        (epoch < 1000    and epoch % 100 == 0)  or 
                        (epoch < 10'000  and epoch % 1000 == 0) or
                        (epoch < 100'000 and epoch % 10'000 == 0)) {
                        std::println("NNet::train: Epoch: {}", epoch);
                    }
                    size_t num_learnings{0};
                    for (const auto &sample : training_samples) {
                        //std::println("Epoch: {}, inputs: {}, targets: {}", epoch, sample.inputs, sample.targets);
                        feed(in_neurons, sample.inputs);
                        num_learnings += learn(out_neurons, sample.targets);
                    }
                    converged = (num_learnings == training_samples.size());
                }
                timer.stop();
                if (converged) {
                    _trained = true;
                    std::println("NNet::train: Training summary:");
                    //for (const auto &[idx, conn] : _connections | std::views::enumerate) {
                    //    std::println("NNet::train: weight[{}]: {}", idx, conn->weight);
                    //}
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
                //std::println("Infer from data: {}", data);
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
            // Learn from target data
            template <NeuronView View>
                bool learn(View out_neurons, const OutputData &targets) {
                    if (std::ranges::distance(out_neurons) != targets.size()) {
                        throw std::invalid_argument("NNet::learn: size error!");
                    }
                    bool learnt{false};            
                    for (const auto &[neuron, target] : std::views::zip(out_neurons, targets)) {
                        learnt = neuron->learn(target);
                        if (not learnt) {
                            break;
                        }
                    }            
                    return learnt;
                }
            // Data members
            bool _trained{false};
            std::vector<std::unique_ptr<Neuron>> _neurons{};
            std::vector<std::unique_ptr<Neuron::SynapticConn>> _connections{};
    };
}