#pragma once
#include <concepts>

namespace hnnet {
    template <typename Backend, typename View, typename DeviceData, typename HostData>
        concept BackendType = requires(Backend &backend,
                                       View &view,
                                       DeviceData &device_data,
                                       HostData &host_data) {
                                  { backend.init(view) } -> std::same_as<void>;
                                  { backend.to_device(view, device_data) } -> std::same_as<void>;
                                  { backend.to_host(view, host_data) } -> std::same_as<void>;
                              };
}
