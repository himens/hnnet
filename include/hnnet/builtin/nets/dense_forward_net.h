#pragma once
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include "hnnet/builtin/activations/identity.h"
#include "hnnet/builtin/nets/dag_net.h"

namespace hnnet::builtin::net {
    template <typename T>
        concept DenseLayerType = requires(T layer) {
            { auto(layer.size) } -> std::same_as<int_t>;
            { auto(layer.type) } -> std::same_as<NeuronType>;
            { auto(layer.activation) } -> ActivationType;
        };
    template <ActivationType Activation>
        struct DenseLayer {
            int_t size;
            NeuronType type;
            Activation activation;
            bool biased{false};
        };
    ///////////////////////////
    // DenseForwardNet class //
    ///////////////////////////
    class DenseForwardNet : public DAGNet {
        public:
            // Constructor
            template <DenseLayerType... Layers>
                requires (sizeof...(Layers) > 0)
                explicit DenseForwardNet(const Layers &...layers) {
                    if (layers...[0].type != NeuronType::input) {
                        throw std::invalid_argument("DenseForwardNet::DenseForwardNet: first layer must be of input type!");
                    }
                    if (layers...[sizeof...(Layers) - 1].type != NeuronType::output) {
                        throw std::invalid_argument("DenseForwardNet::DenseForwardNet: last layer must be of output type!");
                    }
                    auto tx_neurons = this->new_neurons(layers...[0].size, layers...[0].type, layers...[0].activation);
                    auto connect_layer = [&] (const auto &layer) {
                        const auto rx_neurons = this->new_neurons(layer.size, layer.type, layer.activation);
                        this->connect(tx_neurons, rx_neurons);
                        if (layer.biased) {
                            const auto bias = this->new_neurons(1, NeuronType::bias, activation::Identity{});
                            this->connect(bias, rx_neurons);
                        }
                        tx_neurons = rx_neurons;
                    };
                    [&]<size_t... I>(std::index_sequence<I...>) {
                        (connect_layer(layers...[I + 1]), ...);
                    }(std::make_index_sequence<sizeof...(Layers) - 1>{});
                }
    };
}
