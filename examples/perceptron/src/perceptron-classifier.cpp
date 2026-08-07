#include <fstream>
#include "perceptron-neuron.h"
#include "hnnet-nnet.h"

constexpr size_t nb_rows{9};
constexpr size_t nb_columns{7};
constexpr size_t nb_letters{7};

struct LetterData {
    char value;
    hNNet::Data<nb_rows * nb_columns> pixels;
};
// Encode letter
hNNet::Data<nb_letters> encode_letter(const char letter) {
    const auto letter_bit = static_cast<size_t>(letter - 'A');
    if (letter_bit >= nb_letters) {
        throw std::runtime_error("read_letters: invalid letter " + std::string{letter} + " in dataset!");
    }
    hNNet::Data<nb_letters> bits{};
    for (size_t bit{0}; bit < bits.size(); bit++) {
        bits[bit] = (bit == letter_bit) ? +1.0 : -1.0;
    }
    return bits;
}
// Decode letter
char decode_letter(const hNNet::Data<nb_letters> &bits) {
    size_t letter_bit{0};
    for (size_t bit{0}; bit < bits.size(); bit++) {
        if (bits[bit] == +1.0) {
            letter_bit = bit;
            break;
        }
    }
    return static_cast<char>(letter_bit + 'A');
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
            // read letter
            const auto letter_str = line | std::views::split(' ') | std::views::drop(1) | std::views::join | std::ranges::to<std::string>();
            letter.value = letter_str.front();
            // read grid
            size_t pixel_bit{0};
            for (size_t i{0}; i < nb_rows; i++) {
                if (not std::getline(file, line)) {
                    break;
                }
                const auto validated = (line.size() == nb_columns) and std::ranges::all_of(line, [] (const auto &ch) { return ch == '#' or ch == '.'; });
                if (not validated) {
                    throw std::runtime_error("read_letters: invalid grid row: " + line);
                }
                for (const auto &ch : line) {
                    letter.pixels[pixel_bit++] = ch == '#' ? 1.0 : -1.0;  
                }
            }
            letters.push_back(letter);
            //std::println("read_letter: letter: {}, inputs: {}, outputs: {}", letter, data.inputs, data.outputs);
        }
    }
    if (letters.empty()) {
        std::println("read_line: no letter found!");
    }
    return letters;
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
    std::vector<LetterData> letters{};
    letters.append_range(read_letters("data/letters_train_1.txt"));
    letters.append_range(read_letters("data/letters_train_2.txt"));
    letters.append_range(read_letters("data/letters_train_3.txt"));
    std::vector<Classifier::TrainingData> training_samples(letters.size());
    for (const auto &letter : letters) {
        training_samples.push_back({.inputs = letter.pixels, .targets = encode_letter(letter.value)});
    }
    classifier.train(training_samples);
    // use net to classify other similar data (inference)
    for (const auto &letter : read_letters("data/letters_noisy_3.txt")) {
        const auto outputs = classifier.infer(letter.pixels);
        std::println("Expected letter = {}, classifier response = {}", letter.value, decode_letter(outputs));
    }

    return 0;
}