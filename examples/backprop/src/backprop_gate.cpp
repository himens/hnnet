#include "hnnet/builtin/dense_forward_net.h"
#include "hnnet/builtin/backprop_rule.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    using namespace hNNet;
    // create net
    Builtin::DenseForwardNet gate{
        Builtin::Layer{2, NeuronType::input,  Builtin::IdentityActivation{}},
        Builtin::Layer{4, NeuronType::hidden, Builtin::SigmoidActivation{}, true},
        Builtin::Layer{1, NeuronType::output, Builtin::SigmoidActivation{}, true}
    };
    // train net
    const std::array<input_vector_t, 4> inputs{{
        // binary
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0}
        // bipolar
        //{+1, +1},
        //{+1, -1},
        //{-1, +1},
        //{-1, -1}
    }};
    const std::array<output_vector_t, 4> targets{{
        {0}, 
        {1}, 
        {1}, 
        {0}
    }};
    gate.train(inputs, targets, Builtin::BackpropRule{0.2});

    return 0;
}
