#pragma once
#include <concepts>

namespace hnnet {
    template <typename Backend, typename View, typename DeviceData, typename HostData>
        concept BackendType = requires(Backend &backend,
                                       View &view,
                                       HostData &host_data,
                                       DeviceData &device_data) {
                                  { backend.init(view) } -> std::same_as<void>;
                                  { backend.to_device(host_data) } -> std::same_as<void>;
                                  { backend.to_host(device_data) } -> std::same_as<void>;
                              };
}
