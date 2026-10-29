#pragma once
#include <concepts>
#include "hnnet/backend_type.h"

namespace hnnet::net::update {
    template <typename Backend, typename View, typename DeviceData, typename HostData>
        concept BackendType = hnnet::BackendType<Backend, View, DeviceData, HostData> and
                              requires(Backend &backend) {
                                  { backend.update() } -> std::same_as<void>;
                              };
}
