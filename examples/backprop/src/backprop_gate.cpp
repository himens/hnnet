#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/dense_forward_net.h"
#include "hnnet/builtin/backprop_rule.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    using namespace hNNet;
    // create net
    using Gate = Builtin::DenseForwardNet<Data<real_t, 2>, Data<real_t, 1>>;
    Gate gate{
        Builtin::Layer{2, NeuronType::input,  Builtin::IdentityActivation{}},
        Builtin::Layer{4, NeuronType::hidden, Builtin::SigmoidActivation{}, true},
        Builtin::Layer{1, NeuronType::output, Builtin::SigmoidActivation{}, true}
    };
    // train net
    std::vector<Gate::TrainingData> samples{
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
    gate.train(samples, Builtin::BackpropRule{0.2});

    return 0;
}
