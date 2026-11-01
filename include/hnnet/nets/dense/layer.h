#pragma once
#include <concepts>
#include <type_traits>
#include "hnnet/neuron.h"

namespace hnnet::dense {
    template <ActivationType Activation>
        struct Layer {
            int_t size;
            NeuronType type;
            Activation activation;
            bool biased{false};
        };
    template <typename T>
        struct IsLayer : std::false_type {};
    template <ActivationType Activation>
        struct IsLayer<Layer<Activation>> : std::true_type {};
    template <typename T>
        concept LayerType = IsLayer<std::remove_cvref_t<T>>::value;
}
