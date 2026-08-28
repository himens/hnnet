#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/backprop-rule.h"
#include "hnnet/nnet.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    // define net type
    using namespace hNNet;
    using Gate = NNet<Data<real_t, 2>, Data<real_t, 1>>;
    // create net
    Gate gate;
    auto input_layer  = gate.new_neurons(2, NeuronType::input,  Builtin::IdentityActivation{});
    auto hidden_layer = gate.new_neurons(4, NeuronType::hidden, Builtin::SigmoidActivation{});
    auto output_layer = gate.new_neurons(1, NeuronType::output, Builtin::SigmoidActivation{});
    gate.connect(input_layer, hidden_layer);
    gate.connect(hidden_layer, output_layer);
    gate.add_bias(hidden_layer);
    gate.add_bias(output_layer);
    // train net
    std::vector<Gate::TrainingSample> samples = {
        // binary
        {{1, 1}, {0}},
        {{1, 0}, {1}},
        {{0, 1}, {1}},
        {{0, 0}, {0}}
        // bipolar
        //{{+1, +1}, {-1}},
        //{{+1, -1}, {+1}},
        //{{-1, +1}, {+1}},
        //{{-1, -1}, {-1}}
    };
    gate.randomize_weights(-0.5, +0.5);
    gate.train(samples, Builtin::BackpropRule{0.2});

    return 0;
}