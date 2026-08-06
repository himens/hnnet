#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    using GateData = hNNet::TrainData<3, 1>;
    hNNet::NNet gate;
    const auto layer_x = gate.new_neurons<Perceptron::Neuron>(3);
    const auto y = gate.new_neuron<Perceptron::Neuron>();
    gate.connect(layer_x, y);
    std::vector<GateData> training_sample = {
        {{1, 1, 1},  {1}},
        {{1, 0, 1}, {-1}},
        {{0, 1, 1}, {-1}},
        {{0, 0, 1}, {-1}}
    };
    gate.train(training_sample);

    return 0;
}