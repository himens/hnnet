#include "hnnet/builtin/backprop-rule.h"
#include "hnnet/nnet.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    // define net type
    using namespace hNNet;
    using Gate = NNet<Neuron, Data<real_t, 7>, Data<real_t, 1>>;
    // create net
    Gate gate;
    const auto input_layer  = gate.new_neurons(2, IdentityActivation{});
    const auto hidden_layer = gate.new_neurons(4, SigmoidActivation{});
    const auto output_layer = gate.new_neurons(1, SigmoidActivation{});
    gate.connect(input_layer, hidden_layer);
    gate.connect(hidden_layer, output_layer);
    const auto hidden_biases = gate.new_neurons(4, IdentityActivation{});
    for (const auto &[bias, neuron] : std::views::zip(hidden_biases, hidden_layer)) {
        gate.connect(bias, neuron);
    }
    const auto output_bias = gate.new_neurons(1, IdentityActivation{});
    for (const auto &[bias, neuron] : std::views::zip(output_bias, output_layer)) {
        gate.connect(bias, neuron);
    }
    // train net
    std::vector<Gate::TrainingSample> training_samples = {
        // binary
        {{1, 1, 1, 1, 1, 1, 1}, {0}},
        {{1, 0, 1, 1, 1, 1, 1}, {1}},
        {{0, 1, 1, 1, 1, 1, 1}, {1}},
        {{0, 0, 1, 1, 1, 1, 1}, {0}}
        // bipolar
        //{{+1, +1, +1, +1, +1, +1, +1}, {-1}},
        //{{+1, -1, +1, +1, +1, +1, +1}, {+1}},
        //{{-1, +1, +1, +1, +1, +1, +1}, {+1}},
        //{{-1, -1, +1, +1, +1, +1, +1}, {-1}}
    };
    gate.randomize_weights(-0.5, +0.5);
    Builtin::BackpropRule rule;
    gate.train(training_samples, rule);

    return 0;
}