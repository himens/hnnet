#include <fstream>
#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

using ClassifierData = hNNet::TrainData<63, 7>; // grid of 7x9 pixels (63) and 7 letters (A-G)

// Read letter dataset from file
std::vector<ClassifierData> read_letters(const std::string &filename) {
    std::ifstream input{filename};
    if (not input.is_open()) {
        throw std::runtime_error("read_letters: unable to open letter dataset: " + filename + "!");
    }
    // read file and fill train sample
    std::vector<ClassifierData> sample;
    std::string line;
    while (std::getline(input, line)) {
        ClassifierData data{};
        auto letter = line | std::views::take(1) | std::ranges::to<std::string>();
        auto values = line | std::views::split(' ') | std::views::drop(1);
        if (std::ranges::distance(values) != data.inputs.size()) {
            throw std::runtime_error("read_letters: inconsistent number of input features!");
        }
        auto inputs = values | std::views::transform([] (auto range) { return std::string_view(range) == "1" ? 1 : 0; });
        std::ranges::copy(inputs, data.inputs.begin());
        const auto index = static_cast<size_t>(letter[0] - 'A');
        if (index >= data.targets.size()) {
            throw std::runtime_error("read_letters: invalid letter in dataset!");
        }
        data.targets[index] = 1.0;
        sample.push_back(data);
        //std::println("read_letter: letter: {}, inputs: {}, targets: {}", letter, data.inputs, data.targets);
    }
    return sample;
}

// Classify letters using the trained perceptron neural network
int main() {
    hNNet::NNet classifier;
    const auto layer_in = classifier.new_neurons<Perceptron::Neuron>(63);
    const auto layer_out = classifier.new_neurons<Perceptron::Neuron>(7);
    classifier.connect(layer_in, layer_out);
    std::vector<ClassifierData> training_sample{};
    const auto sample1 = read_letters("data/letters_train_1.dat"); 
    const auto sample2 = read_letters("data/letters_train_2.dat");
    const auto sample3 = read_letters("data/letters_train_3.dat");
    training_sample.append_range(sample1);
    training_sample.append_range(sample2);
    training_sample.append_range(sample3);
    classifier.train(training_sample);

    return 0;
}