#pragma once
#include "hnnet/nnet.h"
#include "hnnet/builtin/activations.h"

namespace hNNet::Builtin {
    template <typename T>
        concept LayerType = requires(T layer) {
            { auto(layer.size) } -> std::same_as<int_t>;
            { auto(layer.type) } -> std::same_as<NeuronType>;
            { auto(layer.activation) } -> ActivationType;
        };
    template <ActivationType Activation>
        struct Layer {
            int_t size;
            NeuronType type;
            Activation activation;
        };
    ///////////////////////////
    // DenseForwardNet class //
    ///////////////////////////
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
                            tx_neurons = rx_neurons;
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
