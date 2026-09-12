#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/perceptron_rule.h"
#include "hnnet/nnet.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    using namespace hNNet;
    // create net
    NNet gate;
    auto input_layer  = gate.new_neurons(2, NeuronType::input,  Builtin::IdentityActivation{});
    auto output_layer = gate.new_neurons(1, NeuronType::output, Builtin::PerceptronActivation{});
    auto bias         = gate.new_neurons(1, NeuronType::bias,   Builtin::IdentityActivation{});
    gate.connect(input_layer, output_layer);
    gate.connect(bias, output_layer);
    // train net
    Dataset inputs{
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0}
    };
    Dataset targets{
        {+1}, 
        {-1}, 
        {-1}, 
        {-1}};
    gate.train(inputs, targets, Builtin::PerceptronRule{1.0});

    return 0;
}
