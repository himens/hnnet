#pragma once
#include <utility>
#include "hnnet/nnet.h"
#include "hnnet/builtin/activations.h"

namespace hNNet::Builtin {
    /////////////////
    // LayerType   //
    /////////////////
    // A layer exposes size (int_t), type (NeuronType) and a valid activation object.
    template <typename T>
        concept LayerType = requires(const T &layer) {
            requires std::same_as<std::remove_cvref_t<decltype(layer.size)>, int_t>;
            requires std::same_as<std::remove_cvref_t<decltype(layer.type)>, NeuronType>;
            requires ActivationType<std::remove_cvref_t<decltype(layer.activation)>>;
        };
    /////////////////
    // Layer struct //
    /////////////////
    // Ready-to-use layer descriptor (aggregate, supports CTAD)
    template <ActivationType Activation>
        struct Layer {
            int_t size;
            NeuronType type;
            Activation activation;
        };
    //////////////////////////////
    // DenseForwardNet class    //
    //////////////////////////////
    // Fully-connected feed-forward net: layer[i] is fully connected to layer[i + 1], in order.
    // If the first layer is not of input type, an implicit input layer (size = InputData) is created.
    // The last layer must be of output type.
    template <DataType InputData, DataType OutputData>
        class DenseForwardNet : public NNet<InputData, OutputData> {
            public:
                // Constructor
                template <LayerType... Layers>
                    requires (sizeof...(Layers) > 0)
                    explicit DenseForwardNet(const Layers &...layers) {
                        if (layers...[sizeof...(Layers) - 1].type != NeuronType::output) {
                            throw std::invalid_argument("DenseForwardNet::DenseForwardNet: last layer must be of output type!");
                        }
                        const bool has_input_layer{layers...[0].type == NeuronType::input};
                        // if the first layer is not of input type, create an implicit input layer sized like InputData
                        auto tx_neurons = has_input_layer ?
                            this->new_neurons(layers...[0].size, layers...[0].type, layers...[0].activation) :
                            this->new_neurons(this->input_size, NeuronType::input, IdentityActivation{});
                        auto connect_layer = [&] (const auto &layer) {
                            auto rx_neurons = this->new_neurons(layer.size, layer.type, layer.activation);
                            this->connect(tx_neurons, rx_neurons);
                            tx_neurons = rx_neurons; // rx diventa il tx della prossima connessione
                        };
                        if (has_input_layer) {
                            [&]<size_t... I>(std::index_sequence<I...>) {
                                (connect_layer(layers...[I + 1]), ...);
                            }(std::make_index_sequence<sizeof...(Layers) - 1>{});
                        }
                        else {
                            [&]<size_t... I>(std::index_sequence<I...>) {
                                (connect_layer(layers...[I]), ...);
                            }(std::make_index_sequence<sizeof...(Layers)>{});
                        }
                    }
        };
}
