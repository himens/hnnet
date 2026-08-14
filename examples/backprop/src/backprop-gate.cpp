#include "hnnet/builtin/backprop-neuron.h"
#include "hnnet/nnet.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    // define net type
    using InputData  = hNNet::Data<hNNet::real_t, 7>;
    using OutputData = hNNet::Data<hNNet::real_t, 1>;
    using Gate = hNNet::NNet<InputData, OutputData>;
    using BackpropNeuron = hNNet::Builtin::BackpropNeuron<hNNet::Builtin::SigmoidActivation>;
    // create net
    Gate gate;
    const auto input_layer  = gate.new_neurons<BackpropNeuron>(2);
    const auto hidden_layer = gate.new_neurons<BackpropNeuron>(4);
    const auto output_layer = gate.new_neurons<BackpropNeuron>(1);
    gate.connect(input_layer, hidden_layer);
    gate.connect(hidden_layer, output_layer);
    const auto hidden_biases = gate.new_neurons<BackpropNeuron>(4);
    for (const auto &[bias, neuron] : std::views::zip(hidden_biases, hidden_layer)) {
        gate.connect(bias, neuron);
    }
    const auto output_bias = gate.new_neurons<BackpropNeuron>(1);
    for (const auto &[bias, neuron] : std::views::zip(output_bias, output_layer)) {
        gate.connect(bias, neuron);
    }
    // train net
    std::vector<Gate::TrainingData> training_samples = {
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
    gate.train(training_samples);

    return 0;
}