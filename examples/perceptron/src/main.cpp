#include "hnnet-nnet.h"
class PerceptronNeuron : public hNNet::Neuron {
        // Update conenction weight according to perceptron learning rule
        void update(SynapticConn* conn, const hNNet::real_t target) {
            static constexpr hNNet::real_t rate{1.0};
            conn->weight += rate * target * conn->tx->get_signal();
        }
        // Activation function
        hNNet::real_t activation(const hNNet::real_t weighted_sum) const {
            static constexpr hNNet::real_t threshold{0.2};
            return weighted_sum > threshold ? +1 : weighted_sum < -threshold ? -1 : 0;
        }
};

int main() {
    hNNet::NNet net;
    const auto layer_x = net.new_neurons<PerceptronNeuron>(3);
    const auto y = net.new_neurons<PerceptronNeuron>(1);
    net.connect<PerceptronNeuron, PerceptronNeuron>(layer_x, y);
    
    // Example training data (3 inputs, 1 target) for a simple AND gate
    std::vector<hNNet::TrainData<3, 1>> training_data = {
        {{1, 1, 1},  {1}},
        {{1, 0, 1}, {-1}},
        {{0, 1, 1}, {-1}},
        {{0, 0, 1}, {-1}}
    };
    net.train(training_data);
    
    return 0;
}