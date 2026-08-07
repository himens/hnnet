#include <fstream>
#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

constexpr size_t num_rows{9};
constexpr size_t num_columns{7};
using ClassifierData = hNNet::TrainData<num_rows * num_columns, 7>; // grid with 7 letters (A-G)

// Read letter dataset from file (bipolar inputs, bipolar targets)
std::vector<ClassifierData> read_letters(const std::string &filename) {
    std::ifstream file{filename};
    if (not file.is_open()) {
        throw std::runtime_error("read_letters: unable to open letter dataset: " + filename + "!");
    }
    // read file and fill train sample
    std::vector<ClassifierData> sample;
    std::string line;
    while (std::getline(file, line)) {
        ClassifierData data{};
        if (line.contains("Letter")) {
            // read letter
            const auto letter = line | std::views::split(' ') | std::views::drop(1) | std::views::join | std::ranges::to<std::string>();
            const auto letter_bit = static_cast<size_t>(letter[0] - 'A');
            if (letter_bit >= data.targets.size()) {
                throw std::runtime_error("read_letters: invalid letter " + letter + " in dataset!");
            }
            for (size_t bit{0}; bit < data.targets.size(); bit++) {
                data.targets[bit] = (bit == letter_bit) ? +1.0 : -1.0;
            }
            // read grid
            size_t grid_bit{0};
            for (size_t i{0}; i < num_rows; i++) {
                if (not std::getline(file, line)) {
                    break;
                }
                const auto validated = (line.size() == num_columns) and std::ranges::all_of(line, [] (const auto ch) { return ch == '#' or ch == '.'; });
                if (not validated) {
                    throw std::runtime_error("read_letters: invalid grid row: " + line);
                }
                for (const auto ch : line) {
                    data.inputs[grid_bit++] = ch == '#' ? 1.0 : -1.0;  
                }
            }
            sample.push_back(data);
            //std::println("read_letter: letter: {}, inputs: {}, targets: {}", letter, data.inputs, data.targets);
        }
    }
    if (sample.empty()) {
        std::println("read_line: no letter found!");
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
    training_sample.append_range(read_letters("data/letters_train_1.txt"));
    training_sample.append_range(read_letters("data/letters_train_2.txt"));
    training_sample.append_range(read_letters("data/letters_train_3.txt"));
    classifier.train(training_sample);

    return 0;
}