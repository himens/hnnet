#include <fstream>
#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

constexpr size_t nb_rows{9};
constexpr size_t nb_columns{7};
constexpr size_t nb_pixels{nb_rows * nb_columns};
constexpr size_t nb_letters{26};

using Pixels = std::array<std::array<char, nb_columns>, nb_rows>;
struct LetterData {
    char character{'\0'};
    Pixels pixels{};
};
using InputData  = hNNet::Data<hNNet::int_t, nb_pixels>;
using OutputData = hNNet::Data<hNNet::int_t, nb_letters>;

// Encode pixel grid
InputData encode(const Pixels &pixels) {
    InputData data{};
    std::ranges::fill(data, -1);
    for (const auto &[row, row_pixels] : pixels | std::views::enumerate) {
        for (const auto &[col, pixel] : row_pixels | std::views::enumerate) {
            const auto idx = row * nb_columns + col;
            data[idx] = pixel == '#' ? +1 : -1;
        }
    }
    return data;
}
// Encode letters
OutputData encode(const std::initializer_list<char> &letters) {
    OutputData data;
    std::ranges::fill(data, -1);
    for (const auto &letter : letters) {
        if (letter < 'A' or letter > 'Z') {
            throw std::invalid_argument("encode: invalid letter: " + std::string{letter});
        }
        const auto idx = static_cast<int>(letter - 'A');
        data[idx] = +1;
    }
    return data;
}
// Decode letters
std::vector<char> decode(const OutputData &data) {
    std::vector<char> letters{};
    for (size_t idx{0}; idx < data.size(); idx++) {
        if (data[idx] == +1) {
            const auto letter = static_cast<char>(idx + 'A');
            letters.push_back(letter);
        }
    }
    return letters;
}
// Read letter dataset from file (bipolar inputs, bipolar outputs)
std::vector<LetterData> read_letters(const std::string &filename) {
    std::ifstream file{filename};
    if (not file.is_open()) {
        throw std::runtime_error("read_letters: unable to open letter dataset: " + filename + "!");
    }
    // read file and fill train sample
    std::vector<LetterData> letters;
    std::string line;
    while (std::getline(file, line)) {
        LetterData letter{};
        if (line.contains("Letter")) {
            // read letter character
            letter.character = (line | std::views::split(' ') | std::views::drop(1) | std::views::join | std::ranges::to<std::string>()).front();
            // read pixels
            for (size_t row{0}; row < nb_rows; row++) {
                if (not std::getline(file, line)) {
                    break;
                }
                const auto valid = (line.size() == nb_columns) and std::ranges::all_of(line, [] (const auto &ch) { return ch == '#' or ch == '.'; });
                if (not valid) {
                    throw std::runtime_error("read_letters: invalid pixels row: " + line);
                }
                for (const auto &[col, ch] : line | std::views::enumerate) {
                    letter.pixels[row][col] = ch;
                }
            }
            letters.push_back(letter);
            //std::println("read_letter: letter: {}, inputs: {}, outputs: {}", letter, data.inputs, data.outputs);
        }
    }
    if (letters.empty()) {
        std::println("read_letters: no letter found!");
    }
    return letters;
}

// Classify letters using the trained perceptron neural network
int main() {
    // define net type
    using Classifier = hNNet::NNet<InputData, OutputData>;
    // create net
    Classifier classifier;
    const auto layer_in  = classifier.new_neurons<Perceptron::Neuron>(nb_pixels);
    const auto layer_out = classifier.new_neurons<Perceptron::Neuron>(nb_letters);
    classifier.connect(layer_in, layer_out);
    // train net
    std::vector<LetterData> letters{};
    letters.append_range(read_letters("data/letters_train_1.txt"));
    letters.append_range(read_letters("data/letters_train_2.txt"));
    letters.append_range(read_letters("data/letters_train_3.txt"));
    std::vector<Classifier::TrainingData> training_samples(letters.size());
    for (const auto &[ch, pixels] : letters) {
        training_samples.push_back({.inputs = encode(pixels), .targets = encode({ch})});
    }
    classifier.train(training_samples);
    // use net (inference)
    for (const auto &[ch, pixels] : read_letters("data/letters_noisy_1.txt")) {
        const auto outputs = classifier.infer(encode(pixels));
        std::println("Expected letter = {}, classifier response = {}", ch, decode(outputs));
    }

    return 0;
}