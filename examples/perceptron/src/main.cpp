#include "hnnet-nnet.h"

////////////////////////////
// PerceptronNeuron class //
////////////////////////////
class PerceptronNeuron : public hNNet::Neuron {
    private:
        // Update connection weight according to perceptron learning rule
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
    // Simple AND gate implementation using a perceptron neural network
    hNNet::NNet gate;
    const auto layer_x = gate.new_neurons<PerceptronNeuron>(3);
    const auto y = gate.new_neuron<PerceptronNeuron>();
    gate.connect(layer_x, y);
    std::vector<hNNet::TrainData<3, 1>> training_data = {
        {{1, 1, 1},  {1}},
        {{1, 0, 1}, {-1}},
        {{0, 1, 1}, {-1}},
        {{0, 0, 1}, {-1}}
    };
    gate.train(training_data);
    
    return 0;
}