#pragma once
#include <vector>
#include <span>
#include <memory>
#include "hnnet-neuron.h"

namespace hNNet {
    ////////////////
    // NNet class //
    ////////////////
    class NNet {
        public:
            // Create new neuron
            Neuron* new_neuron() {
                auto neuron = std::make_unique<Neuron>();
                const auto raw_ptr = neuron.get();
                _neurons.push_back(std::move(neuron));
                return raw_ptr;
            }
            // Create new neurons
            std::vector<Neuron*> new_neurons(const size_t num_neurons) {
                std::vector<Neuron*> neurons(num_neurons);
                for (size_t i{0}; i < num_neurons; ++i) {
                    neurons[i] = new_neuron();
                }
                return neurons;
            }
            // Connect neurons
            void connect(const std::span<Neuron*> tx_neurons, const std::span<Neuron*> rx_neurons) {
                auto neurons = _neurons | std::views::transform([] (const auto &neuron) { return neuron.get(); });
                if (not std::ranges::includes(neurons, tx_neurons) or not std::ranges::includes(neurons, rx_neurons)) {
                    throw std::invalid_argument("NNet::connect: neurons not in network!");
                }
                for (auto &[tx, rx] : std::views::zip(tx_neurons, rx_neurons)) {
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
                        for (const auto &data : sample)) {
                            feed(data.input);
                            converged = backprop(data.target);  
                        }     
                    }
                }  
        private:
            // Perfrorm
            template <size_t Size>
                void feed(const Data<Size> &data) {
                    auto input_layer = _neurons 
                            | std::view::transform([] (const auto &neuron) { return neuron->get_in_connections().empty(); });
                    if (std::ranges::distance(input_layer) != Size) {
                        throw std::invalid_argument("NNet::feed: size error!");
                    }
                    for (auto &[neuron, input] : std::view:zip(input_layer, data)) {
                        neuron->set_signal(data.input);
                        neuron->broadcast_signal();
                }
            // Perform back-propagation
            template <size_t Size>
                bool backprop(const Data<Size> &data) {
                    auto output_layer = _neurons // static?
                            | std::views::transform([] (const auto &neuron) { return neuron->get_out_connections().empty(); });
                    if (std::ranges::distance(output_layer) != Size) {
                        throw std::invalid_argument("NNet::backprop: size error!");
                    }
                    bool learnt{false};            
                    for (auto &[neuron, target] : std::views::zip(output_layer, data)) {
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
    }

}
