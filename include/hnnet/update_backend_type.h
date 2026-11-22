#pragma once
#include <concepts>
#include "hnnet/backend_type.h"

namespace hnnet::net::update {
    template <typename Backend, typename HostData, typename DeviceData, typename View, typename Weights = std::span<const real_t>>
        concept BackendType = hnnet::BackendType<Backend, HostData, DeviceData, View> and
                              requires(Backend &backend, const Weights &weights) {
                                  { backend.update_states(weights) } -> std::same_as<void>;
                              };
}
