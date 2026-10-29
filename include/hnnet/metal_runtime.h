#pragma once
#if !defined(__APPLE__) || !defined(__OBJC__)
#error "hnnet Metal requires macOS and an Objective-C++ translation unit"
#else
#if !__has_feature(objc_arc)
#error "hnnet Metal requires Objective-C ARC (-fobjc-arc)"
#endif
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace hnnet::metal {
    // Check that a count fits Metal kernel indexing
    inline std::uint32_t checked_count(const std::size_t count) {
        if (count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("Metal: tensor exceeds 32-bit kernel indexing");
        }
        return static_cast<std::uint32_t>(count);
    }

    // Check that a product fits Metal kernel indexing
    inline std::size_t checked_product(const std::size_t first, const std::size_t second) {
        if (second != 0 and first > std::numeric_limits<std::uint32_t>::max() / second) {
            throw std::length_error("Metal: tensor exceeds 32-bit kernel indexing");
        }
        return first * second;
    }

    // Describe a Metal error
    inline std::string error_message(NSError *error) {
        if (error == nil) return "unspecified Metal error";
        const auto message = [[error localizedDescription] UTF8String];
        return message == nullptr ? "unspecified Metal error" : message;
    }

    class Context {
        public:
            // Initialize the device, command queue and shader library
            void init(const std::string_view source) {
                _device = MTLCreateSystemDefaultDevice();
                if (_device == nil) throw std::runtime_error("Metal: no compatible device");
                _queue = [_device newCommandQueue];
                if (_queue == nil) throw std::runtime_error("Metal: cannot create command queue");
                NSString *text = [[NSString alloc] initWithBytes:source.data()
                                                       length:source.size()
                                                     encoding:NSUTF8StringEncoding];
                if (text == nil) throw std::runtime_error("Metal: kernel source is not UTF-8");
                MTLCompileOptions *options = [MTLCompileOptions new];
                options.mathMode = MTLMathModeFast;
                NSError *error = nil;
                _library = [_device newLibraryWithSource:text options:options error:&error];
                if (_library == nil) {
                    throw std::runtime_error("Metal: kernel compilation failed: " + error_message(error));
                }
            }

            // Allocate a shared buffer, optionally clearing its contents
            id<MTLBuffer> buffer(const std::size_t bytes, const bool clear = true) const {
                const auto length = std::max(bytes, sizeof(float));
                if (length > [_device maxBufferLength]) {
                    throw std::length_error("Metal: buffer exceeds device maxBufferLength");
                }
                id<MTLBuffer> result = [_device newBufferWithLength:length options:MTLResourceStorageModeShared];
                if (result == nil) throw std::runtime_error("Metal: buffer allocation failed");
                if (clear) std::memset([result contents], 0, length);
                return result;
            }

            // Upload host values into a shared buffer
            template <typename T>
                id<MTLBuffer> upload(const std::vector<T> &values) const {
                    id<MTLBuffer> result = buffer(values.size() * sizeof(T), false);
                    if (not values.empty()) {
                        std::memcpy([result contents], values.data(), values.size() * sizeof(T));
                    }
                    return result;
                }

            // Get the Metal device
            id<MTLDevice> device() const {
                return _device;
            }

            // Create a compute pipeline from a library function
            id<MTLComputePipelineState> pipeline(const char *name) const {
                id<MTLFunction> function = [_library newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (function == nil) throw std::runtime_error(std::string{"Metal: missing kernel "} + name);
                NSError *error = nil;
                id<MTLComputePipelineState> result = [_device newComputePipelineStateWithFunction:function error:&error];
                if (result == nil) {
                    throw std::runtime_error(std::string{"Metal: cannot create pipeline "} + name + ": " + error_message(error));
                }
                return result;
            }

            // Create a command buffer on the context queue
            id<MTLCommandBuffer> command() const {
                id<MTLCommandBuffer> result = [_queue commandBuffer];
                if (result == nil) throw std::runtime_error("Metal: cannot create command buffer");
                return result;
            }

        private:
            id<MTLDevice> _device = nil;
            id<MTLCommandQueue> _queue = nil;
            id<MTLLibrary> _library = nil;
    };

    // Create an encoder for a compute pipeline
    inline id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> command,
                                               id<MTLComputePipelineState> pipeline) {
        id<MTLComputeCommandEncoder> result = [command computeCommandEncoder];
        if (result == nil) throw std::runtime_error("Metal: cannot create compute encoder");
        [result setComputePipelineState:pipeline];
        return result;
    }

    // Dispatch a compute grid and finish encoding
    inline void dispatch(id<MTLComputeCommandEncoder> encoder,
                         id<MTLComputePipelineState> pipeline,
                         const std::size_t count) {
        if (count != 0) {
            const auto width = std::min<NSUInteger>([pipeline maxTotalThreadsPerThreadgroup], 256);
                [encoder dispatchThreadgroups:MTLSizeMake((count + width - 1) / width, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        }
        [encoder endEncoding];
    }

    // Submit a command buffer and check its completion
    inline void complete(id<MTLCommandBuffer> command) {
        [command commit];
        [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Metal: command failed: " + error_message([command error]));
        }
    }
}
#endif
