#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/perceptron-rule.h"
#include "hnnet/nnet.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    // define net type
    using namespace hNNet;
    using Gate = NNet<Data<int_t, 2>, Data<int_t, 1>>;
    // create net
    Gate gate;
    auto input_layer  = gate.new_neurons(2, NeuronType::input,  Builtin::IdentityActivation{});
    auto output_layer = gate.new_neurons(1, NeuronType::output, Builtin::PerceptronActivation{});
    auto bias =         gate.new_neurons(1, NeuronType::bias,   Builtin::IdentityActivation{});
    gate.connect(input_layer, output_layer);
    gate.connect(bias, output_layer);
    // train net
    std::vector<Gate::TrainingSample> samples = {
        {{1, 1},  {1}},
        {{1, 0}, {-1}},
        {{0, 1}, {-1}},
        {{0, 0}, {-1}}
    };
    gate.train(samples, Builtin::PerceptronRule{1.0});

    return 0;
}
