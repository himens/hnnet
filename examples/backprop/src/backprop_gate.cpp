#include <array>
#include "hnnet/builtin/activations/identity.h"
#include "hnnet/builtin/activations/sigmoid.h"
#include "hnnet/builtin/losses/mse.h"
#include "hnnet/builtin/nets/dense_forward_net.h"
#include "hnnet/builtin/backends/omp/backprop/backprop.h"
#include "hnnet/builtin/rules/backprop.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    using namespace hnnet;
    // create net
    builtin::net::DenseForwardNet gate{
        builtin::net::DenseLayer{2, NeuronType::input,  builtin::activation::Identity{}},
        builtin::net::DenseLayer{4, NeuronType::hidden, builtin::activation::Sigmoid{}, true},
        builtin::net::DenseLayer{1, NeuronType::output, builtin::activation::Sigmoid{}, true}
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
    gate.train(
        inputs,
        targets,
        builtin::rule::Backprop{
            builtin::backend::omp::backprop::Momentum{0.2},
            builtin::loss::MSE{},
            1
    });

    return 0;
}
