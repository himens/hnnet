#include <array>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/perceptron.h"
#include "hnnet/nets/dag_net.h"
#include "hnnet/rules/perceptron.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    using namespace hnnet;
    // create net
    net::DAGNet gate;
    auto input_layer  = gate.new_neurons(2, NeuronType::input,  activation::Identity{});
    auto output_layer = gate.new_neurons(1, NeuronType::output, activation::Perceptron{});
    auto bias         = gate.new_neurons(1, NeuronType::bias,   activation::Identity{});
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
    gate.train(inputs, targets, rule::Perceptron{1.0_real});

    return 0;
}
