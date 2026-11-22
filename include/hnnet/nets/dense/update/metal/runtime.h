#pragma once
#include "hnnet/metal_runtime.h"
#include <bit>
#include <cmath>
#include <filesystem>
#include <ranges>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "activations.h"

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
    static_assert(sizeof(DenseLayer) == 20);

    // Assemble the shaders shared by standalone update and backprop
    inline std::string shader_source(const std::filesystem::path &directory) {
        return activation_shader(activation_sources(activation_directory(directory))) + "\n"
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

    // Pack the neurons' activations in the layout of the generated ActivationDescriptor
    inline std::vector<std::uint32_t> describe_activations(const std::ranges::input_range auto &neurons, const std::filesystem::path &directory) {
        const auto sources = activation_sources(activation_directory(directory));
        const auto stride = descriptor_stride(sources);
        std::vector<std::uint32_t> result;
        for (const auto &neuron : neurons) {
            const auto description = describe_activation(neuron.activation());
            const auto source = std::ranges::find(sources, description.name, &ActivationSource::name);
            if (source == sources.end() or source->fields.size() != description.fields.size()) {
                throw std::invalid_argument("Metal: no matching header for activation " + std::string{description.name});
            }
            result.push_back(static_cast<std::uint32_t>(source - sources.begin()));
            for (const auto &field : source->fields) {
                const auto value = std::ranges::find(description.fields, field, &std::pair<std::string_view, real_t>::first);
                if (value == description.fields.end()) {
                    throw std::invalid_argument("Metal: no value for " + source->name + "::" + field);
                }
                result.push_back(std::bit_cast<std::uint32_t>(to_float(value->second)));
            }
            result.resize(result.size() + stride - 1 - source->fields.size(), 0);
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
