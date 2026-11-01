#pragma once
#include <cstddef>
#include <stdexcept>
#include <utility>
#include "hnnet/activations/identity.h"
#include "hnnet/nets/dag/dag_net.h"
#include "hnnet/nets/dense/layer.h"
#include "hnnet/topology.h"

namespace hnnet::net {
    ///////////////////////////
    // DenseForwardNet class //
    ///////////////////////////
    class DenseForwardNet final : public DAGNet {
        public:
            class View : public DAGNet::View {
                private:
                    friend class DenseForwardNet;
                    // Construct a dense-forward view.
                    explicit View(DenseForwardNet &net) : DAGNet::View(net) {}
            };
            // Constructor
            template <dense::LayerType... Layers>
                requires (sizeof...(Layers) > 0)
                explicit DenseForwardNet(const Layers &...layers) : DAGNet(make_topology(layers...)) {}
            // Create a dense-forward view of the network.
            View view() {
                return View(*this);
            }
        private:
            // Make dense topology
            template <dense::LayerType... Layers>
                requires (sizeof...(Layers) > 0)
                static Topology make_topology(const Layers &...layers) {
                    if (layers...[0].type != NeuronType::input) {
                        throw std::invalid_argument("DenseForwardNet::DenseForwardNet: first layer must be of input type!");
                    }
                    if (layers...[sizeof...(Layers) - 1].type != NeuronType::output) {
                        throw std::invalid_argument("DenseForwardNet::DenseForwardNet: last layer must be of output type!");
                    }
                    Topology topology;
                    auto tx_indices = topology.new_neurons(layers...[0].size, layers...[0].type, layers...[0].activation);
                    auto connect_layer = [&] (const auto &layer) {
                        const auto bias_indices = layer.biased ? topology.new_neurons(1, NeuronType::bias, activation::Identity{}) : std::vector<index_t>{};
                        const auto rx_indices = topology.new_neurons(layer.size, layer.type, layer.activation);
                        topology.connect(tx_indices, rx_indices);
                        if (layer.biased) {
                            topology.connect(bias_indices, rx_indices);
                        }
                        tx_indices = rx_indices;
                    };
                    [&]<size_t... LayerIndex>(std::index_sequence<LayerIndex...>) {
                        (connect_layer(layers...[LayerIndex + 1]), ...);
                    }(std::make_index_sequence<sizeof...(Layers) - 1>{});
                    return topology;
                }
    };
}
