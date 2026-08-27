#include <fstream>
#include <sstream>
#include "hnnet/activation.h"
#include "hnnet/builtin/backprop-rule.h"
#include "hnnet/nnet.h"

constexpr size_t nb_pixels{784};
constexpr size_t nb_classes{10};
constexpr size_t nb_hidden{128};
constexpr size_t nb_training_samples{60000};
constexpr size_t nb_test_samples{0};
constexpr size_t batch_size{nb_training_samples};

using namespace hNNet;
struct DigitData {
    int_t label{0};
    std::array<int_t, nb_pixels> pixels{};
};
using InputData  = Data<real_t, nb_pixels>;
using OutputData = Data<real_t, nb_classes>;

// Encode pixel grid (grayscale [0, 255] -> normalized [0.0, 1.0])
InputData encode(const std::array<int_t, nb_pixels> &pixels) {
    InputData data{};
    for (const auto &[idx, pixel] : pixels | std::views::enumerate) {
        data[idx] = static_cast<real_t>(pixel) / 255.0;
    }
    return data;
}
// Encode digit label (one-hot)
OutputData encode(const int_t label) {
    OutputData data{};
    std::ranges::fill(data, 0.0);
    if (label < 0 or label >= static_cast<int_t>(nb_classes)) {
        throw std::invalid_argument("encode: invalid label: " + std::to_string(label));
    }
    data[label] = 1.0;
    return data;
}
// Decode digit label (index of the highest activation)
int_t decode(const OutputData &data) {
    const auto it = std::ranges::max_element(data);
    return std::distance(data.begin(), it);
}
// Read MNIST dataset from a CSV file (each row: label,pixel0,pixel1,...,pixel783)
std::vector<DigitData> read_digits(const std::string &filename, const size_t max_samples) {
    std::ifstream file{filename};
    if (not file.is_open()) {
        throw std::runtime_error("read_digits: unable to open MNIST dataset: " + filename + "!");
    }
    std::vector<DigitData> digits;
    std::string line;
    while ((digits.size() < max_samples) and std::getline(file, line)) {
        std::istringstream stream{line};
        std::string token;
        DigitData digit{};
        std::getline(stream, token, ',');
        digit.label = std::stoll(token);
        for (size_t idx{0}; idx < nb_pixels; idx++) {
            if (not std::getline(stream, token, ',')) {
                throw std::runtime_error("read_digits: invalid row (missing pixels): " + line);
            }
            digit.pixels[idx] = std::stoll(token);
        }
        digits.push_back(digit);
    }
    if (digits.empty()) {
        std::println("read_digits: no digit found!");
    }
    return digits;
}
// Classify MNIST handwritten digits using a back-propagation neural network w/ one hidden layer
int main() {
    // define net type
    using Classifier = NNet<InputData, OutputData>;
    // create net
    Classifier classifier;
    auto input_layer  = classifier.new_neurons(nb_pixels, NeuronType::input);
    auto hidden_layer = classifier.new_neurons(nb_hidden, NeuronType::hidden, SigmoidActivation{});
    auto output_layer = classifier.new_neurons(nb_classes, NeuronType::output, SigmoidActivation{});
    classifier.connect(input_layer, hidden_layer);
    classifier.connect(hidden_layer, output_layer);
    // read train and test samples
    const auto train_digits = read_digits("data/mnist/mnist_train.csv", nb_training_samples);
    const auto test_digits = read_digits("data/mnist/mnist_test.csv", nb_test_samples);
    std::vector<Classifier::TrainingSample> samples;
    samples.reserve(nb_training_samples);
    for (const auto &[label, pixels] : train_digits) {
        samples.push_back({.inputs = encode(pixels), .targets = encode(label)});
    }
    // train net
    classifier.randomize_weights(-0.1, 0.1);
    classifier.train(samples, Builtin::BackpropRule{0.05, 0.9});
    // eval efficiency
    auto eval_efficiency = [&] (const std::vector<DigitData> &digits) {
        size_t error_count{0};
        for (const auto &[label, pixels] : digits) {
            const auto outputs = classifier.infer(encode(pixels));
            const auto predicted_label = decode(outputs);
            error_count += (predicted_label != label);
        }
        return 100.0 * (1.0 - static_cast<real_t>(error_count) / digits.size());
    };
    std::println("MNIST classification efficiency (train) = {:.2f}%", eval_efficiency(train_digits));
    std::println("MNIST classification efficiency (test) = {:.2f}%", eval_efficiency(test_digits));

    return 0;
}
