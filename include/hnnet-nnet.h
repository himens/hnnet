#pragma once
#include "hnnet-neuron.h"

namespace hNNet {
    ////////////////
    // NNet class //
    ////////////////
    class NNet {
        public:
            // Create new neurons
            std::vector<Neuron*> new_neurons(const size_t num_neurons) {
                std::vector<Neuron*> neurons(num_neurons);
                for (size_t i{0}; i < num_neurons; ++i) {
                    auto neuron = std::make_unique<Neuron>();
                    neurons[i] = neuron.get();
                    _neurons.push_back(std::move(neuron));
                }
                return neurons;
            }
            // Connect neurons
            void connect(std::span<Neuron* const> tx_neurons, std::span<Neuron* const> rx_neurons) {
                auto neurons = _neurons | std::views::transform([] (const auto &neuron) { return neuron.get(); });
                if (not std::ranges::includes(neurons, tx_neurons) or not std::ranges::includes(neurons, rx_neurons)) {
                    throw std::invalid_argument("NNet::connect: neurons not in network!");
                }
                for (const auto &[tx, rx] : std::views::zip(tx_neurons, rx_neurons)) {
                    auto conn = std::make_unique<Neuron::SynapticConn>(tx, rx);
                    tx->add_connection(conn.get());
                    rx->add_connection(conn.get());
                    _connections.push_back(std::move(conn));
                }
            }
            // Train net
            template <size_t SizeInput, size_t SizeTarget>
                void train(const std::vector<TrainData<SizeInput, SizeTarget>> &sample) {
                    bool converged{false};
                    while (not converged) { 
                        for (const auto &data : sample) {
                            feed(data.input);
                            converged = backprop(data.target);  
                        }     
                    }
                }  
        private:
            // Feed net with input data
            template <size_t Size>
                void feed(const Data<Size> &data) {
                    auto input_layer = _neurons 
                            | std::views::filter([] (const auto &neuron) { return neuron->get_in_connections().empty(); });
                    if (std::ranges::distance(input_layer) != Size) {
                        throw std::invalid_argument("NNet::feed: size error!");
                    }
                    for (auto [neuron, input] : std::views::zip(input_layer, data)) {
                        neuron->set_signal(input);
                        neuron->broadcast_signal();
                    }
                }
            // Perform back-propagation
            template <size_t Size>
                bool backprop(const Data<Size> &data) {
                    auto output_layer = _neurons // static?
                            | std::views::filter([] (const auto &neuron) { return neuron->get_out_connections().empty(); });
                    if (std::ranges::distance(output_layer) != Size) {
                        throw std::invalid_argument("NNet::backprop: size error!");
                    }
                    bool learnt{false};            
                    for (auto [neuron, target] : std::views::zip(output_layer, data)) {
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
