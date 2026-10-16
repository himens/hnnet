#include <array>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense_forward_net.h"
#include "hnnet/rules/backprop/backprop.h"
#include "hnnet/rules/backprop/omp/backend.h"
#include "hnnet/rules/backprop/omp/optimizers/momentum.h"

// Simple XOR gate implementation using a back-propagation neural network w/ one hidden layer
int main() {
    using namespace hnnet;
    // create net
    net::DenseForwardNet gate{
        net::DenseLayer{2, NeuronType::input,  activation::Identity{}},
        net::DenseLayer{4, NeuronType::hidden, activation::Sigmoid{}, true},
        net::DenseLayer{1, NeuronType::output, activation::Sigmoid{}, true}
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
        rule::Backprop{
            backprop::omp::Backend<backprop::omp::optimizer::Momentum>{0.2},
            loss::MSE{},
            1
    });

    return 0;
}
