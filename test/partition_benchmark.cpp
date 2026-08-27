#include "hnnet/nnet.h"

#include <chrono>

using namespace hNNet;

namespace {
    constexpr size_t input_count{256};
    constexpr size_t hidden_count{4096};
    constexpr size_t output_count{256};
  
    using Net = NNet<Data<real_t, input_count>, Data<real_t, output_count>>;

    struct NoOpLearningRule {
        template <typename Network>
        void learn(Network&, const output_t<Network>&) {}
    };
}

int main() {
    Net net;
    const auto inputs = net.new_neurons(input_count, NeuronType::input);
    const auto hidden = net.new_neurons(hidden_count, NeuronType::hidden);
    const auto hidden2 = net.new_neurons(hidden_count, NeuronType::hidden);
    const auto outputs = net.new_neurons(output_count, NeuronType::output);

    net.connect(inputs, hidden);
    net.connect(hidden, hidden2);
    net.connect(hidden2, outputs);

    Net::TrainingSample sample{};
    std::vector<Net::TrainingSample> samples(100, sample);
    net.train(samples, NoOpLearningRule{});
}
