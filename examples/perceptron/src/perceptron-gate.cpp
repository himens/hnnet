#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    // define net type
    using InputData  = hNNet::Data<3, hNNet::int_t>;
    using OutputData = hNNet::Data<1, hNNet::int_t>;
    using Gate = hNNet::NNet<InputData, OutputData>;
    // create net
    Gate gate;
    const auto layer_x = gate.new_neurons<Perceptron::Neuron>(3);
    const auto y = gate.new_neuron<Perceptron::Neuron>();
    gate.connect(layer_x, y);
    // tain net
    std::vector<Gate::TrainingData> training_samples = {
        {{1, 1, 1},  {1}},
        {{1, 0, 1}, {-1}},
        {{0, 1, 1}, {-1}},
        {{0, 0, 1}, {-1}}
    };
    gate.train(training_samples);

    return 0;
}