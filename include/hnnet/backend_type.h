#pragma once
#include <concepts>

namespace hnnet {
    template <typename Backend, typename HostData, typename DeviceData, typename ...InitArgs>
        concept BackendType = requires(Backend &backend,
                                       HostData &data,
                                       DeviceData &result,
                                       InitArgs &...args) {
                                  { backend.init(args...) } -> std::same_as<void>;
                                  { backend.to_device(data) } -> std::same_as<void>;
                                  { backend.to_host(result) } -> std::same_as<void>;
                              };
}
