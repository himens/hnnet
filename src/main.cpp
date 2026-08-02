#include "hnnet.h"

int main() {
    hNNet::NNet net;
    auto neuron1 = net.new_neuron();
    auto neuron2 = net.new_neuron();
    //net.add_connection(std::move(conn));
    
    // Example training data
    std::vector<hNNet::TrainData<1, 1>> training_data = {
        {{0.0}, {0.0}},
        {{1.0}, {1.0}}
    };
    
    net.train(training_data);
    
    return 0;
}