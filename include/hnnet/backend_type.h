#pragma once
#include <concepts>
#include "hnnet/types.h"

namespace hnnet {
    template <typename Backend, typename View, typename... Data>
        concept BackendType = requires(Backend &backend,
                                       View &view,
                                       const Data &...data) {
                                  { backend.init(view) } -> std::same_as<void>;
                                  { backend.to_device(view, data...) } -> std::same_as<void>;
                                  { backend.to_host(view) } -> std::same_as<void>;
                              };
}
