#include <fstream>
#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

constexpr size_t nb_rows{9};
constexpr size_t nb_columns{7};
constexpr size_t nb_letters{7};

struct LetterGrid {
    char letter;
    std::array<std::array<char, nb_columns>, nb_rows> pixels;
};
// Encode pixels
hNNet::Data<nb_rows * nb_columns> encode_pixels(const std::array<std::array<char, nb_columns>, nb_rows> &pixels) {
    return {};
}
// Encode letters
hNNet::Data<nb_letters> encode_letters(const std::vector<char> &letters) {
    //if (bit >= nb_letters) {
    //    throw std::runtime_error("read_letters: invalid letter " + std::string{letter} + " in dataset!");
    //}
    hNNet::Data<nb_letters> bits;
    std::ranges::fill(bits, -1);
    for (const auto &letter : letters) {
        const auto bit = static_cast<int>(letter - 'A');
        bits[bit] = +1;
    }
    return bits;
}
// Decode letters
std::vector<char> decode(const hNNet::Data<nb_letters> &bits) {
    std::vector<char> letters{};
    for (size_t bit{0}; bit < bits.size(); bit++) {
        if (bits[bit] == +1) {
            letters.push_back(static_cast<char>(bit + 'A'));
            break;
        }
    }
    return letters;
}
// Read letter dataset from file (bipolar inputs, bipolar outputs)
std::vector<LetterGrid> read_letters(const std::string &filename) {
    std::ifstream file{filename};
    if (not file.is_open()) {
        throw std::runtime_error("read_letters: unable to open letter dataset: " + filename + "!");
    }
    // read file and fill train sample
    std::vector<LetterGrid> grids;
    std::string line;
    while (std::getline(file, line)) {
        LetterGrid grid{};
        if (line.contains("Letter")) {
            // read letter
            const auto letter = line | std::views::split(' ') | std::views::drop(1) | std::views::join | std::ranges::to<std::string>();
            grid.letter = letter.front();
            // read pixels
            for (size_t row{0}; row < nb_rows; row++) {
                if (not std::getline(file, line)) {
                    break;
                }
                const auto validated = (line.size() == nb_columns) and std::ranges::all_of(line, [] (const auto &ch) { return ch == '#' or ch == '.'; });
                if (not validated) {
                    throw std::runtime_error("read_letters: invalid grid row: " + line);
                }
                for (const auto &[col, ch] : line | std::views::enumerate) {
                    grid.pixels[row][col] = ch;
                }
            }
            grids.push_back(grid);
            //std::println("read_letter: letter: {}, inputs: {}, outputs: {}", letter, data.inputs, data.outputs);
        }
    }
    if (grids.empty()) {
        std::println("read_line: no letter found!");
    }
    return grids;
}

// Classify letters using the trained perceptron neural network
int main() {
    // create net
    using Classifier = hNNet::NNet<nb_rows * nb_columns, nb_letters>; 
    Classifier classifier;
    const auto layer_in  = classifier.new_neurons<Perceptron::Neuron>(63);
    const auto layer_out = classifier.new_neurons<Perceptron::Neuron>(7);
    classifier.connect(layer_in, layer_out);
    // train net
    std::vector<LetterGrid> grids{};
    grids.append_range(read_letters("data/letters_train_1.txt"));
    grids.append_range(read_letters("data/letters_train_2.txt"));
    grids.append_range(read_letters("data/letters_train_3.txt"));
    std::vector<Classifier::TrainingData> training_samples(grids.size());
    for (const auto &[letter, pixels] : grids) {
        training_samples.push_back({.inputs = encode_pixels(pixels), .targets = encode_letters({letter})});
    }
    classifier.train(training_samples);
    // use net to classify other similar data (inference)
    for (const auto &[letter, pixels] : read_letters("data/letters_noisy_3.txt")) {
        const auto outputs = classifier.infer(encode_pixels(pixels));
        std::println("Expected letter = {}, classifier response = {}", letter, decode(outputs));
    }

    return 0;
}