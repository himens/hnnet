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
#include <utility>
#include <vector>
#include "hnnet/activations/identity.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/activations/relu.h"
#include "hnnet/losses/mse.h"
#include "hnnet/nets/dense_forward/dense_forward_net.h"
#include "hnnet/nets/dag/update/omp/backend.h"
#include "hnnet/rules/backprop/backprop.h"
#ifdef METAL
#include "hnnet/rules/backprop/metal/backend.h"
#include "hnnet/rules/backprop/metal/optimizers/momentum.h"
#else
#include "hnnet/rules/backprop/omp/backend.h"
#include "hnnet/rules/backprop/omp/optimizers/momentum.h"
#endif

// Constants
constexpr size_t nb_training_samples{60'000};
constexpr size_t nb_test_samples{10'000};
constexpr size_t nb_pixels{784};
constexpr size_t nb_digits{10};
// Aliases and data types
using namespace hnnet;
using Pixels = std::array<int_t, nb_pixels>;
struct DigitData {
    int_t value{0};
    Pixels pixels{};
};
// Encode digits to training data
TrainingData<dataset_vector_t> encode(const std::vector<DigitData> &digits) {
    // Encode pixels (grayscale [0, 255] -> normalized [0.0, 1.0])
    auto encode_pixels = [] (const Pixels &pixels) {
        std::vector<real_t> data(nb_pixels, 0.0_real);
        for (const auto &[ipx, pixel] : pixels | std::views::enumerate) {
            data[ipx] = static_cast<real_t>(pixel) / 255.0_real;
        }
        return data;
    };
    // Encode digit value
    auto encode_value = [] (const int_t value) {
        std::vector<real_t> data(nb_digits, 0.0_real);
        if (value < 0 or value >= static_cast<int_t>(nb_digits)) {
            throw std::invalid_argument("encode: invalid value: " + std::to_string(value));
        }
        data[value] = 1.0_real;
        return data;
    };
    TrainingData<dataset_vector_t> data;
    data.inputs.resize(nb_training_samples);
    data.targets.resize(nb_training_samples);
    for (const auto &[digit_index, digit] : digits | std::views::enumerate) {
        data.inputs[digit_index]  = encode_pixels(digit.pixels);
        data.targets[digit_index] = encode_value(digit.value);
    }
    return data;
}
// Decode digit value (index of the highest activation)
int_t decode(const DataType auto &data) {
    const auto it = std::ranges::max_element(data);
    return std::distance(data.begin(), it);
}
// Read MNIST dataset from a CSV file (each row: value,pixel0,pixel1,...,pixel783)
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
        digit.value = std::stoll(token);
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
// Eval efficiency
real_t eval_efficiency(const dataset_vector_t &outputs, const dataset_vector_t &targets) {
    size_t error_count{0};
    for (const auto &[output, target] : std::views::zip(outputs, targets)) {
        const auto digit = decode(target);
        const auto predicted_digit = decode(output);
        error_count += (predicted_digit != digit);
    }
    return 100.0_real * (1.0_real - static_cast<real_t>(error_count) / targets.size());
};
// Run OpenMP benchmark
#ifndef METAL
void omp_benchmark(const std::vector<DigitData> &train_digits, const std::vector<DigitData> &test_digits) {
    std::println("==================");
    std::println("OpenMP benchmark: ");
    std::println("==================");
    // create net
    constexpr size_t nb_hidden{128};
    net::DenseForwardNet classifier{
        net::DenseLayer{nb_pixels, NeuronType::input,  activation::Identity{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::Sigmoid{}},
        net::DenseLayer{nb_digits, NeuronType::output, activation::Sigmoid{}}
    };
    // encode data
    const auto train_data = encode(train_digits);
    const auto test_data = encode(test_digits);
    // train
    classifier.train(
        train_data,
        rule::Backprop{
            backprop::omp::Backend<backprop::omp::optimizer::Momentum>{0.05_real, 0.9_real},
            loss::MSE{},
            1
        });
    // eval efficiency
    const auto train_outputs = classifier.infer(train_data.inputs, dag::update::omp::Backend{});
    const auto test_outputs = classifier.infer(test_data.inputs, dag::update::omp::Backend{});
    std::println("MNIST classification efficiency (train) = {:.2f}%", eval_efficiency(train_outputs, train_data.targets));
    std::println("MNIST classification efficiency (test) = {:.2f}%", eval_efficiency(test_outputs, test_data.targets));
}
#endif
// Run Metal benchmark
#ifdef METAL
void metal_benchmark(const std::vector<DigitData> &train_digits, const std::vector<DigitData> &test_digits) {
    std::println("==================");
    std::println("Metal benchmark:  ");
    std::println("==================");
    // create net
    constexpr size_t nb_hidden{1024};
    net::DenseForwardNet classifier{
        net::DenseLayer{nb_pixels, NeuronType::input,  activation::Identity{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_hidden, NeuronType::hidden, activation::ReLU{}},
        net::DenseLayer{nb_digits, NeuronType::output, activation::Sigmoid{}}
    };
    // encode data
    const auto train_data = encode(train_digits);
    const auto test_data = encode(test_digits);
    // train net
    classifier.train(
        train_data,
        rule::Backprop{
            backprop::metal::Backend<backprop::metal::optimizer::Momentum>{0.3_real, 0.5_real},
            loss::MSE{},
            1024
    });
    // eval efficiency
    const auto train_outputs = classifier.infer(train_data.inputs, dense_forward::update::metal::Backend{});
    const auto test_outputs = classifier.infer(test_data.inputs, dense_forward::update::metal::Backend{});
    std::println("MNIST classification efficiency (train) = {:.2f}%", eval_efficiency(train_outputs, train_data.targets));
}
#endif
// Classify MNIST handwritten digits using a back-propagation neural network w/ one hidden layer
int main() {
    const auto train_digits = read_digits("data/mnist/mnist_train.csv", nb_training_samples);
    const auto test_digits = read_digits("data/mnist/mnist_test.csv", nb_test_samples);
#ifdef METAL
    metal_benchmark(train_digits, test_digits);
#else
    omp_benchmark(train_digits, test_digits);
#endif
    return 0;
}
