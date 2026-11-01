#pragma once
#include <concepts>
#include "hnnet/backend_type.h"

namespace hnnet::net::update {
    template <typename Backend, typename View, typename HostData, typename DeviceData>
        concept BackendType = hnnet::BackendType<Backend, HostData, DeviceData, View> and
                              requires(Backend &backend, View &view) {
                                  { backend.update_states(view.weights()) } -> std::same_as<void>;
                              };
}
