#pragma once
#include "hnnet/nets/dense_forward/update/metal/runtime.h"

namespace hnnet::backprop::metal::detail {
    using hnnet::metal::checked_count;
    using hnnet::metal::checked_product;
    using hnnet::metal::complete;
    using hnnet::metal::dispatch;
    using hnnet::metal::encoder;
    using hnnet::metal::error_message;

    using dense_forward::update::metal::detail::Parameters;
    using dense_forward::update::metal::detail::ActivationDescriptor;
    using dense_forward::update::metal::detail::DenseLayer;
    using dense_forward::update::metal::detail::read_source;
    using dense_forward::update::metal::detail::to_float;
    using dense_forward::update::metal::detail::describe_activation;
    using dense_forward::update::metal::detail::matrix;
    using dense_forward::update::metal::detail::product;

    struct UpdateParameters {
        std::uint32_t weight_count;
        float first;
        float second;
    };

    static_assert(sizeof(Parameters) == 28);
    static_assert(sizeof(ActivationDescriptor) == 8);
    static_assert(sizeof(UpdateParameters) == 12);

    class Context : public hnnet::metal::Context {
        public:
            // Assemble the backprop shaders and initialize the common context
            void init(const std::filesystem::path &directory, const std::filesystem::path &update_directory) {
                const auto source = dense_forward::update::metal::detail::shader_source(update_directory)
                    + read_source(directory / "kernels.metal");
                hnnet::metal::Context::init(source);
            }
    };
}
