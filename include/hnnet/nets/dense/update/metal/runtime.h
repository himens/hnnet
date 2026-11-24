#pragma once
#include "hnnet/metal_runtime.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "hnnet/activation.h"

namespace hnnet::dense::update::metal::detail {
    struct Parameters {
        std::uint32_t neuron_count;
        std::uint32_t connection_count;
        std::uint32_t input_count;
        std::uint32_t output_count;
        std::uint32_t output_begin;
        std::uint32_t batch_begin;
        std::uint32_t batch_size;
    };

    struct DenseLayer {
        std::uint32_t tx_begin;
        std::uint32_t tx_count;
        std::uint32_t rx_begin;
        std::uint32_t rx_count;
        std::uint32_t weight_offset;
    };

    static_assert(sizeof(Parameters) == 28);
    static_assert(sizeof(real_t) == sizeof(float), "Metal: Activation is uploaded as is and requires FP32 real_t");
    static_assert(sizeof(DenseLayer) == 20);

    // Read a shader source file
    inline std::string read_source(const std::filesystem::path &path) {
        std::ifstream stream{path, std::ios::binary};
        if (not stream) throw std::runtime_error("Metal: cannot read " + path.string());
        return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    }

    // Read a header shared with the CPU: activation includes are inlined, any other include is dropped
    inline std::string read_shared_source(const std::filesystem::path &include_directory, const std::filesystem::path &file) {
        constexpr std::string_view prefix{"#include \"hnnet/activations/"};
        std::istringstream stream{read_source(include_directory / file)};
        std::string result;
        for (std::string line; std::getline(stream, line);) {
            if (line.starts_with(prefix)) {
                result += read_shared_source(include_directory, line.substr(10, line.find('"', 10) - 10));
            }
            else if (not line.starts_with("#include") and not line.starts_with("#pragma once")) {
                result += line + "\n";
            }
        }
        return result;
    }

    // Assemble the shaders shared by standalone update and backprop
    inline std::string shader_source(const std::filesystem::path &directory) {
        return read_source(directory / "activations.h") + "\n"
            + read_shared_source(directory / "../../../../..", "hnnet/activation.h") + "\n"
            + read_source(directory / "kernels.metal") + "\n";
    }

    // Locate the installed update shaders
    inline std::filesystem::path kernel_directory() {
        return std::filesystem::path{__FILE__}.parent_path();
    }

    // Convert host values to finite FP32 values
    inline float to_float(const real_t value) {
        const auto result = static_cast<float>(value);
        if (not std::isfinite(result)) {
            throw std::invalid_argument("Metal: values must be finite and representable in FP32");
        }
        return result;
    }

    // Helper function to create an MPS matrix view from a buffer
    inline MPSMatrix *matrix(
        id<MTLBuffer> buffer,
        const std::uint32_t offset,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t stride) {
        MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                              columns:columns
                                                                             rowBytes:std::size_t{stride} * sizeof(float)
                                                                             dataType:MPSDataTypeFloat32];
        MPSMatrix *result = [[MPSMatrix alloc] initWithBuffer:buffer
                                                     offset:std::size_t{offset} * sizeof(float)
                                                 descriptor:descriptor];
        if (result == nil) throw std::runtime_error("Metal: cannot construct MPS matrix view");
        return result;
    }

    // Helper function to create an MPS matrix multiplication object
    inline MPSMatrixMultiplication *product(
        id<MTLDevice> device,
        const bool transpose_left,
        const bool transpose_right,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t interior,
        const double alpha) {
        MPSMatrixMultiplication *result = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:transpose_left transposeRight:transpose_right
            resultRows:rows resultColumns:columns interiorColumns:interior alpha:alpha beta:0.0];
        if (result == nil) throw std::runtime_error("Metal: cannot construct MPS matrix multiplication");
        return result;
    }
}
