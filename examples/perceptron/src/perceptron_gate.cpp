#include <array>
#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/nets.h"
#include "hnnet/builtin/perceptron.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    using namespace hnnet;
    // create net
    builtin::net::DAGNet gate;
    auto input_layer  = gate.new_neurons(2, NeuronType::input,  builtin::activation::Identity{});
    auto output_layer = gate.new_neurons(1, NeuronType::output, builtin::activation::Perceptron{});
    auto bias         = gate.new_neurons(1, NeuronType::bias,   builtin::activation::Identity{});
    gate.connect(input_layer, output_layer);
    gate.connect(bias, output_layer);
    // train net
    const std::array<input_vector_t, 4> inputs{{
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0}
    }};
    const std::array<output_vector_t, 4> targets{{
        {+1}, 
        {-1}, 
        {-1}, 
        {-1}
    }};
    gate.train(inputs, targets, builtin::perceptron::Rule{1.0});

    return 0;
}
