#include "hnnet/activation.h"
#include "hnnet/builtin/perceptron-rule.h"
#include "hnnet/nnet.h"

// Simple AND gate implementation using a perceptron neural network
int main() {
    // define net type
    using namespace hNNet;
    using Gate = NNet<Neuron, Data<int_t, 3>, Data<int_t, 1>>;
    // create net
    Gate gate;
    const auto layer_x = gate.new_neurons(3, IdentityActivation{});
    const auto y = gate.new_neuron(PerceptronActivation{});
    gate.connect(layer_x, y);
    // train net
    std::vector<Gate::TrainingSample> samples = {
        {{1, 1, 1},  {1}},
        {{1, 0, 1}, {-1}},
        {{0, 1, 1}, {-1}},
        {{0, 0, 1}, {-1}}
    };
    gate.train(samples, Builtin::PerceptronRule{});

    return 0;
}