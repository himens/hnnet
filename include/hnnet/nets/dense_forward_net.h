#pragma once
#include <concepts>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <ranges>
#include "hnnet/activations/identity.h"
#include "hnnet/learning_rule_type.h"
#include "hnnet/nets/dag_net.h"

namespace hnnet::net {
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
    class DenseForwardNet final : public DAGNet {
        public:
            struct Layer {
                index_t tx_begin;
                int_t tx_count;
                index_t rx_begin;
                int_t rx_count;
                index_t weight_offset;
                int_t weight_stride;
                index_t bias_index;
            };
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
                    auto tx_begin = std::ssize(this->neurons());
                    auto tx_indices = this->new_neurons(layers...[0].size, layers...[0].type, layers...[0].activation);
                    auto connect_layer = [&] (const auto &layer) {
                        const auto rx_begin = std::ssize(this->neurons());
                        const auto weight_offset = std::ssize(this->connections());
                        const auto rx_indices = this->new_neurons(layer.size, layer.type, layer.activation);
                        this->connect(tx_indices, rx_indices);
                        index_t bias_index{-1};
                        if (layer.biased) {
                            bias_index = this->new_neurons(1, NeuronType::bias, activation::Identity{}).front();
                            this->connect(std::views::single(bias_index), rx_indices);
                        }
                        _layers.push_back({.tx_begin     = tx_begin, 
                                          .tx_count      = std::ssize(tx_indices),
                                          .rx_begin      = rx_begin, 
                                          .rx_count      = std::ssize(rx_indices),
                                          .weight_offset = weight_offset, 
                                          .weight_stride = std::ssize(tx_indices) + layer.biased,
                                          .bias_index    = bias_index});
                        tx_begin = rx_begin;
                        tx_indices = rx_indices;
                    };
                    [&]<size_t... LayerIndex>(std::index_sequence<LayerIndex...>) {
                        (connect_layer(layers...[LayerIndex + 1]), ...);
                    }(std::make_index_sequence<sizeof...(Layers) - 1>{});
                    init();
                }
            // Train net using a set of training samples
            template <DatasetType Inputs, DatasetType Targets, LearningRuleType<DenseForwardNet, Inputs, Targets> LearningRule>
                void train(const Inputs &inputs, const Targets &targets, LearningRule rule) {
                    NNet::train(*this, inputs, targets, std::move(rule));
                }
            // Get layers
            std::span<const Layer> layers() const {
                return _layers;
            }
        private:
            // Data members
            std::vector<Layer> _layers;
    };
}
