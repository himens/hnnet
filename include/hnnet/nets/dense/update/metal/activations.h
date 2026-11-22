#pragma once
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "hnnet/activations/bipolar_sigmoid.h"
#include "hnnet/activations/bipolar_step.h"
#include "hnnet/activations/elu.h"
#include "hnnet/activations/gaussian.h"
#include "hnnet/activations/identity.h"
#include "hnnet/activations/leaky_relu.h"
#include "hnnet/activations/linear.h"
#include "hnnet/activations/perceptron.h"
#include "hnnet/activations/relu.h"
#include "hnnet/activations/sigmoid.h"
#include "hnnet/activations/softplus.h"
#include "hnnet/activations/softsign.h"
#include "hnnet/activations/step.h"
#include "hnnet/activations/tanh.h"

namespace hnnet::dense::update::metal::detail {
    // A CPU activation class translated for Metal
    struct ActivationSource {
        std::string name;
        std::string member;
        std::vector<std::string> fields;
        std::string source;
    };

    // Host values of an activation, by data member name
    struct ActivationDescription {
        std::string_view name;
        std::vector<std::pair<std::string_view, real_t>> fields;
    };

    // Read a shader source file
    inline std::string read_source(const std::filesystem::path &path) {
        std::ifstream stream{path, std::ios::binary};
        if (not stream) throw std::runtime_error("Metal: cannot read " + path.string());
        return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    }

    // Locate the CPU activation headers from the update shader directory
    inline std::filesystem::path activation_directory(const std::filesystem::path &directory) {
        return directory / "../../../../activations";
    }

    // Rewrite one line of a CPU activation class as Metal code
    inline std::string translate_line(std::string line) {
        static const std::regex literal{R"(([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)_real)"};
        static const std::regex integer{R"([0-9]+)"};
        static const std::regex removed{R"(constexpr | final)"};
        static const std::regex real{R"(\breal_t\b)"};
        static const std::regex absolute{R"(\bstd::abs\b)"};
        static const std::regex standard{R"(\bstd::)"};
        std::string result;
        auto position = line.cbegin();
        for (std::sregex_iterator match{line.cbegin(), line.cend(), literal}; match != std::sregex_iterator{}; ++match) {
            const auto number = (*match)[1].str();
            result.append(position, (*match)[0].first);
            result += number + (std::regex_match(number, integer) ? ".0f" : "f");
            position = (*match)[0].second;
        }
        result.append(position, line.cend());
        result = std::regex_replace(result, removed, "");
        result = std::regex_replace(result, real, "float");
        result = std::regex_replace(result, absolute, "::metal::fabs");
        return std::regex_replace(result, standard, "::metal::");
    }

    // Translate a CPU activation header into a Metal struct
    inline ActivationSource translate_activation(const std::filesystem::path &path) {
        static const std::regex declaration{R"(class (\w+) : public Activation \{)"};
        static const std::regex field{R"(^\s*real_t (\w+)\{)"};
        std::istringstream stream{read_source(path)};
        ActivationSource result{.name = {}, .member = path.stem().string(), .fields = {}, .source = {}};
        auto depth = 0;
        auto closed = false;
        for (std::string line; std::getline(stream, line);) {
            std::smatch match;
            if (result.name.empty()) {
                if (std::regex_search(line, match, declaration)) {
                    result.name = match[1].str();
                    result.source = "    struct " + result.name + " {\n";
                }
                continue;
            }
            if (line.starts_with("    };")) {
                closed = true;
                break;
            }
            if (line.ends_with("public:")) continue;
            if (depth == 0 and std::regex_search(line, match, field)) result.fields.push_back(match[1].str());
            depth += std::ranges::count(line, '{') - std::ranges::count(line, '}');
            result.source += translate_line(line.starts_with("    ") ? line.substr(4) : line) + "\n";
        }
        if (not closed) throw std::runtime_error("Metal: cannot translate " + path.string());
        result.source += "    };\n";
        return result;
    }

    // Translate every CPU activation header; the position in the result is the kind used by the kernels
    inline std::vector<ActivationSource> activation_sources(const std::filesystem::path &directory) {
        std::vector<std::filesystem::path> paths;
        for (const auto &entry : std::filesystem::directory_iterator{directory}) {
            if (entry.path().extension() == ".h") paths.push_back(entry.path());
        }
        if (paths.empty()) throw std::runtime_error("Metal: no activation headers in " + directory.string());
        std::ranges::sort(paths);
        std::vector<ActivationSource> result;
        for (const auto &path : paths) {
            result.push_back(translate_activation(path));
        }
        return result;
    }

    // Count the 32-bit words of the generated ActivationDescriptor
    inline std::size_t descriptor_stride(const std::vector<ActivationSource> &sources) {
        std::size_t field_count{1};
        for (const auto &source : sources) {
            field_count = std::max(field_count, source.fields.size());
        }
        return 1 + field_count;
    }

    // Generate the Metal activation structs, their descriptor and its dispatch
    inline std::string activation_shader(const std::vector<ActivationSource> &sources) {
        std::string structs;
        std::string members;
        std::string values;
        std::string derivatives;
        for (std::size_t kind = 0; kind < sources.size(); ++kind) {
            const auto &source = sources[kind];
            const auto label = "            case " + std::to_string(kind) + ": return descriptor." + source.member;
            structs += source.source;
            members += "            activation::" + source.name + " " + source.member + ";\n";
            values += label + "(value);\n";
            derivatives += label + ".derivative(value);\n";
        }
        return "#include <metal_stdlib>\n"
               "\n"
               "namespace hnnet {\n"
               "namespace activation {\n"
               + structs +
               "}\n"
               "namespace dense {\n"
               "namespace update {\n"
               "namespace metal {\n"
               "    struct ActivationDescriptor {\n"
               "        uint kind;\n"
               "        union {\n"
               + members +
               "        };\n"
               "    };\n"
               "    static_assert(sizeof(ActivationDescriptor) == " + std::to_string(4 * descriptor_stride(sources)) + ", \"activation layout differs from the host\");\n"
               "\n"
               "    inline float activate(const ActivationDescriptor descriptor, const float value) {\n"
               "        switch (descriptor.kind) {\n"
               + values +
               "        }\n"
               "        return 0.0f;\n"
               "    }\n"
               "\n"
               "    inline float derivative(const ActivationDescriptor descriptor, const float value) {\n"
               "        switch (descriptor.kind) {\n"
               + derivatives +
               "        }\n"
               "        return 0.0f;\n"
               "    }\n"
               "}\n"
               "}\n"
               "}\n"
               "}\n";
    }

    // Describe the activation function for Metal kernels
    inline ActivationDescription describe_activation(const Activation *activation) {
        if (const auto bipolar_sigmoid = dynamic_cast<const activation::BipolarSigmoid *>(activation)) {
            return {"BipolarSigmoid", {{"sigma", bipolar_sigmoid->sigma}}};
        }
        if (const auto bipolar_step = dynamic_cast<const activation::BipolarStep *>(activation)) {
            return {"BipolarStep", {{"threshold", bipolar_step->threshold}}};
        }
        if (dynamic_cast<const activation::ELU *>(activation) != nullptr) return {"ELU", {}};
        if (const auto gaussian = dynamic_cast<const activation::Gaussian *>(activation)) {
            return {"Gaussian", {{"mean", gaussian->mean}, {"sigma", gaussian->sigma}}};
        }
        if (dynamic_cast<const activation::Identity *>(activation) != nullptr) return {"Identity", {}};
        if (dynamic_cast<const activation::LeakyReLU *>(activation) != nullptr) return {"LeakyReLU", {}};
        if (const auto linear = dynamic_cast<const activation::Linear *>(activation)) {
            return {"Linear", {{"slope", linear->slope}, {"offset", linear->offset}}};
        }
        if (const auto perceptron = dynamic_cast<const activation::Perceptron *>(activation)) {
            return {"Perceptron", {{"threshold", perceptron->threshold}}};
        }
        if (dynamic_cast<const activation::ReLU *>(activation) != nullptr) return {"ReLU", {}};
        if (const auto sigmoid = dynamic_cast<const activation::Sigmoid *>(activation)) {
            return {"Sigmoid", {{"sigma", sigmoid->sigma}}};
        }
        if (dynamic_cast<const activation::Softplus *>(activation) != nullptr) return {"Softplus", {}};
        if (dynamic_cast<const activation::Softsign *>(activation) != nullptr) return {"Softsign", {}};
        if (const auto step = dynamic_cast<const activation::Step *>(activation)) {
            return {"Step", {{"threshold", step->threshold}}};
        }
        if (dynamic_cast<const activation::Tanh *>(activation) != nullptr) return {"Tanh", {}};
        throw std::invalid_argument("Metal: unsupported activation");
    }
}
