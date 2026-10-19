#include <array>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/perceptron.h"
#include "hnnet/nets/dense_forward_net.h"
#include "hnnet/rules/perceptron.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    using namespace hnnet;
    // create net
    net::DenseForwardNet gate{
        net::DenseLayer{2, NeuronType::input, activation::Identity{}},
        net::DenseLayer{1, NeuronType::output, activation::Perceptron{}, true}
    };
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
