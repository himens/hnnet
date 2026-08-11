#include "backprop-neuron.h"
#include "hnnet-nnet.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    // define net type
    using InputData  = hNNet::Data<hNNet::int_t, 3>;
    using OutputData = hNNet::Data<hNNet::int_t, 1>;
    using Gate = hNNet::NNet<InputData, OutputData>;
    // create net
    Gate gate;
    const auto layer_x = gate.new_neurons<BackpropNeuron>(2);
    const auto layer_h = gate.new_neurons<BackpropNeuron>(4);
    const auto y = gate.new_neuron<BackpropNeuron>();
    gate.connect(layer_x, layer_h);
    gate.connect(layer_h, y);
    // tain net
    std::vector<Gate::TrainingData> training_samples = {
        {{1, 1}, {0}},
        {{1, 0}, {1}},
        {{0, 1}, {1}},
        {{0, 0}, {0}}
    };
    gate.train(training_samples);

    return 0;
}