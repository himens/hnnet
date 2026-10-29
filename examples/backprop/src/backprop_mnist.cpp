#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <print>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/activations/relu.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense_forward/dense_forward_net.h"
#include "hnnet/nets/dag/update/omp/backend.h"
#include "hnnet/rules/backprop/backprop.h"
#include "hnnet/rules/backprop/omp/backend.h"
#include "hnnet/rules/backprop/omp/optimizers/momentum.h"
//#include "hnnet/rules/backprop/metal/backend.h"
//#include "hnnet/rules/backprop/metal/optimizers/momentum.h"

// Constants
constexpr size_t nb_training_samples{60'000};
constexpr size_t nb_test_samples{10'000};
constexpr size_t nb_pixels{784};
constexpr size_t nb_digits{10};
constexpr size_t nb_hidden{512};
// Aliases and data types
using namespace hnnet;
using Pixels = std::array<int_t, nb_pixels>;
struct DigitData {
    int_t label{0};
    Pixels pixels{};
};
// Encode pixel grid (grayscale [0, 255] -> normalized [0.0, 1.0])
input_vector_t encode(const Pixels &pixels) {
    input_vector_t data(nb_pixels, 0.0_real);
    for (const auto &[ipx, pixel] : pixels | std::views::enumerate) {
        data[ipx] = static_cast<real_t>(pixel) / 255.0_real;
    }
    return data;
}
// Encode digit label (one-hot)
output_vector_t encode(const int_t label) {
    output_vector_t data(nb_digits, 0.0_real);
    if (label < 0 or label >= static_cast<int_t>(nb_digits)) {
        throw std::invalid_argument("encode: invalid label: " + std::to_string(label));
    }
    data[label] = 1.0_real;
    return data;
}
// Decode digit label (index of the highest activation)
int_t decode(const DataType auto &data) {
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
    // create net
    net::DenseForwardNet classifier{
        net::DenseLayer{nb_pixels, NeuronType::input,  activation::Identity{}},
        //net::DenseLayer{nb_hidden, NeuronType::hidden, activation::Sigmoid{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        //net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        //net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        //net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_digits, NeuronType::output, activation::Sigmoid{}}
    };
    // read train and test samples
    const auto train_digits = read_digits("data/mnist/mnist_train.csv", nb_training_samples);
    const auto test_digits = read_digits("data/mnist/mnist_test.csv", nb_test_samples);
    std::vector<input_vector_t> inputs(nb_training_samples);
    std::vector<output_vector_t> targets(nb_training_samples);
    for (const auto &[i, digit] : train_digits | std::views::enumerate) {
        inputs[i] = encode(digit.pixels);
        targets[i] = encode(digit.label);
    }
    // train net
    classifier.train(
        inputs,
        targets,
        rule::Backprop{
            //backprop::metal::Backend<backprop::metal::optimizer::Momentum>{0.1_real, 0.6_real},
            backprop::omp::Backend<backprop::omp::optimizer::Momentum>{0.1_real, 0.6_real},
            loss::MSE{},
            512
    });
    //classifier.train(
    //    inputs,
    //    targets,
    //    rule::Backprop{
    //        backprop::omp::Backend<backprop::omp::optimizer::Momentum>{0.05_real, 0.9_real},
    //        loss::MSE{},
    //        1
    //});
    //classifier.train(
    //    inputs,
    //    targets,
    //    rule::Backprop{
    //        backend::omp::backprop::Rprop{1.2, 0.5},
    //        loss::MSE{},
    //        1
    //});
    // eval efficiency
    auto eval_efficiency = [&] (const std::vector<DigitData> &digits) {
        size_t error_count{0};
        for (const auto &[label, pixels] : digits) {
            const auto outputs = classifier.infer(encode(pixels), dag::update::omp::Backend{});
            const auto predicted_label = decode(outputs);
            error_count += (predicted_label != label);
        }
        return 100.0 * (1.0 - static_cast<real_t>(error_count) / digits.size());
    };
    std::println("MNIST classification efficiency (train) = {:.2f}%", eval_efficiency(train_digits));
    std::println("MNIST classification efficiency (test) = {:.2f}%", eval_efficiency(test_digits));

    return 0;
}
