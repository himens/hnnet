#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/perceptron_rule.h"
#include "hnnet/builtin/dag_net.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    using namespace hnnet;
    // create net
    builtin::DAGNet gate;
    auto input_layer  = gate.new_neurons(2, NeuronType::input,  builtin::IdentityActivation{});
    auto output_layer = gate.new_neurons(1, NeuronType::output, builtin::PerceptronActivation{});
    auto bias         = gate.new_neurons(1, NeuronType::bias,   builtin::IdentityActivation{});
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
    gate.train(inputs, targets, builtin::PerceptronRule{1.0});

    return 0;
}
