#include <fstream>
#include "hnnet/builtin/activations.h"
#include "hnnet/builtin/perceptron_rule.h"
#include "hnnet/nnet.h"

// Constants
constexpr size_t nb_rows{9};
constexpr size_t nb_columns{7};
constexpr size_t nb_pixels{nb_rows * nb_columns};
constexpr size_t nb_letters{26};
// Aliases and data types
using namespace hNNet;
using Pixels = std::array<std::array<char, nb_columns>, nb_rows>;
struct LetterData {
    char character{'\0'};
    Pixels pixels{};
};
// Encode pixel grid
input_vector_t encode(const Pixels &pixels) {
    input_vector_t data(nb_pixels, -1);
    for (const auto &[row, row_pixels] : pixels | std::views::enumerate) {
        for (const auto &[col, pixel] : row_pixels | std::views::enumerate) {
            const auto idx = row * nb_columns + col;
            data[idx] = pixel == '#' ? +1 : -1;
        }
    }
    return data;
}
// Encode letters
output_vector_t encode(const std::initializer_list<char> &letters) {
    output_vector_t data(nb_letters, -1);
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
std::vector<char> decode(const DataType auto &data) {
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
        }
    }
    if (letters.empty()) {
        std::println("read_letters: no letter found!");
    }
    return letters;
}
// Classify letters using the trained perceptron neural network
int main() {
    // create net
    NNet classifier;
    auto input_layer  = classifier.new_neurons(nb_rows * nb_columns, NeuronType::input,  Builtin::IdentityActivation{});
    auto output_layer = classifier.new_neurons(nb_letters,           NeuronType::output, Builtin::PerceptronActivation{});
    classifier.connect(input_layer, output_layer);
    // train net
    std::vector<LetterData> letters{};
    letters.append_range(read_letters("data/letters/train_1.txt"));
    letters.append_range(read_letters("data/letters/train_2.txt"));
    letters.append_range(read_letters("data/letters/train_3.txt"));
    std::vector<input_vector_t> inputs(letters.size());
    std::vector<output_vector_t> targets(letters.size());
    for (const auto &[i, letter] : letters | std::views::enumerate) {
        inputs[i] = encode(letter.pixels);
        targets[i] = encode({letter.character});
    }
    classifier.train(inputs, targets, Builtin::PerceptronRule{1.0});
    // use net (inference)
    for (const auto &[ch, pixels] : read_letters("data/letters/noisy_1.txt")) {
        const auto outputs = classifier.infer(encode(pixels));
        std::println("Expected letter = {}, classifier response = {}", ch, decode(outputs));
    }

    return 0;
}
