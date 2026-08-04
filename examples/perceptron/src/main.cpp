#include "hnnet-nnet.h"

int main() {
    hNNet::NNet net;
    const auto layer_x = net.new_neurons(3);
    const auto y = net.new_neurons(1);
    net.connect(layer_x, y);
    
    // Example training data
    std::vector<hNNet::TrainData<3, 1>> training_data = {
        {{1, 1, 1},  {1}},
        {{1, 0, 1}, {-1}},
        {{0, 1, 1}, {-1}},
        {{0, 0, 1}, {-1}}
    };
    net.train(training_data);
    
    return 0;
}