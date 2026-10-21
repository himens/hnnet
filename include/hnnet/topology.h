#pragma once
#include <vector>
#include "hnnet/neuron.h"

namespace hnnet {
    struct Connection {
        index_t tx_index;
        index_t rx_index;
    };  
    struct Topology {
        // Data members
        std::vector<Neuron> neurons;
        std::vector<Connection> connections;
         // Create new neurons
        template <ActivationType Activation>
            index_vector_t new_neurons(const int_t count, const NeuronType &type, const Activation &activation) {
                if (count < 0) {
                    throw std::invalid_argument("NNet::new_neurons: invalid count!");
                }
                for (auto count_index{0}; count_index < count; ++count_index) {
                    neurons.push_back(Neuron{type, std::make_unique<Activation>(activation)});
                }
                return std::views::iota(neurons.size() - count, neurons.size()) | std::ranges::to<index_vector_t>();
            }
        // Append connections; endpoint validation is performed by the network constructor.
        void connect(IndexRange auto &&tx_indices, IndexRange auto &&rx_indices) {
            //for (const auto &[tx_index, rx_index] : std::views::cartesian_product(tx_indices, rx_indices))
            for (const auto &tx_index : tx_indices) {
                for (const auto &rx_index : rx_indices) {
                    connections.push_back({.tx_index = tx_index, .rx_index = rx_index});
                }
            }
        }
        // Connect neurons (zip)
        void zip_connect(IndexRange auto &&tx_indices, IndexRange auto &&rx_indices) {
            auto tx_iterator = std::ranges::begin(tx_indices);
            auto rx_iterator = std::ranges::begin(rx_indices);
            while ((tx_iterator != std::ranges::end(tx_indices)) and (rx_iterator != std::ranges::end(rx_indices))) {
                connect(std::views::single(*tx_iterator), std::views::single(*rx_iterator));
                ++tx_iterator;
                ++rx_iterator;
            }
            if ((tx_iterator != std::ranges::end(tx_indices)) or (rx_iterator != std::ranges::end(rx_indices))) {
                throw std::invalid_argument("NNet::zip_connect: size error!");
            }
        }
    };
}