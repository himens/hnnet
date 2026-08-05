#pragma once
#include "hnnet-neuron.h"

namespace hNNet {
    ////////////////
    // NNet class //
    ////////////////
    class NNet {
        public:            
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
                std::vector<Neuron*> new_neurons(const size_t num_neurons) {
                    if (num_neurons == 0) {
                        throw std::invalid_argument("NNet::new_neurons: num_neurons must be > 0!");
                    }
                    std::vector<Neuron*> neurons(num_neurons);
                    for (auto &neuron : neurons) {
                        neuron = new_neuron<T>();
                    }
                    return neurons;
                }
            // Connect neurons
            template <NeuronPtrRange RangeTx, NeuronPtrRange RangeRx>
                void connect(const RangeTx &tx_range, const RangeRx &rx_range) {
                    auto neurons = _neurons | std::views::transform([] (const auto &neuron) { return neuron.get(); });
                    auto is_in_net = [&] (Neuron *neuron) {
                        return std::ranges::any_of(neurons, [&] (Neuron* net_neuron) { return net_neuron == neuron; });
                    };
                    if (not std::ranges::all_of(tx_range, is_in_net) or not std::ranges::all_of(rx_range, is_in_net)) {
                        throw std::invalid_argument("NNet::connect: neurons not in net!");
                    }
                    for (const auto &[tx, rx] : std::views::cartesian_product(tx_range, rx_range)) {
                        auto conn = std::make_unique<Neuron::SynapticConn>(tx, rx);
                        tx->add_connection(conn.get());
                        rx->add_connection(conn.get());
                        _connections.push_back(std::move(conn));
                    }
                }
            template <NeuronPtrRange Range>
                void connect(const Range &tx_range, Neuron* rx) {
                    connect(tx_range, std::views::single(rx));
                } 
            template <NeuronPtrRange Range>
                void connect(Neuron* tx, const Range &rx_range) {
                    connect(std::views::single(tx), rx_range);
                } 
            void connect(Neuron* tx, Neuron* rx) {
                connect(std::views::single(tx), std::views::single(rx));
            } 
            // Train net
            template <size_t SizeInput, size_t SizeTarget>
                void train(const std::vector<TrainData<SizeInput, SizeTarget>> &sample) {
                    auto in_neurons = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_in_connections().empty(); });
                    auto out_neurons = _neurons | std::views::filter([] (const auto &neuron) { return neuron->get_out_connections().empty(); });
                    constexpr size_t max_epochs{1000};
                    size_t epoch{0};
                    bool converged{false};
                    while (not converged and (epoch <= max_epochs)) {
                        std::println("======================");
                        std::println("Epoch: {}", ++epoch    );
                        std::println("======================");
                        size_t num_learnings{0};
                        for (const auto &data : sample) {
                            std::println("Training with input: {}, target: {}", data.inputs, data.targets);
                            feed(in_neurons, data.inputs);
                            num_learnings += learn(out_neurons, data.targets);
                        }
                        converged = (num_learnings == sample.size());
                    }
                    if (converged) {
                        std::println("NNet::train: net converged successfully! Epochs: {}", epoch);
                    }
                    else {
                        std::println("NNet::train: net training failed!");
                    }
                }  
        private:
            // Feed net with input data
            template <size_t Size>
                void feed(std::ranges::view auto in_neurons, const Data<Size> &inputs) {
                    if (std::ranges::distance(in_neurons) != Size) {
                        throw std::invalid_argument("NNet::feed: size error!");
                    }
                    for (auto [neuron, input] : std::views::zip(in_neurons, inputs)) {
                        neuron->set_signal(input);
                        neuron->broadcast_signal();
                    }
                }
            // Learn from target data
            template <size_t Size>
                bool learn(std::ranges::view auto out_neurons, const Data<Size> &targets) {
                    if (std::ranges::distance(out_neurons) != Size) {
                        throw std::invalid_argument("NNet::learn: size error!");
                    }
                    bool learnt{false};            
                    for (auto [neuron, target] : std::views::zip(out_neurons, targets)) {
                        learnt = neuron->learn(target);
                        if (not learnt) {
                            break;
                        }
                    }            
                    return learnt;
                }
            // Data members
            std::vector<std::unique_ptr<Neuron>> _neurons{};
            std::vector<std::unique_ptr<Neuron::SynapticConn>> _connections{};
    };
}