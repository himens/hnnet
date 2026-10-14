#pragma once
#include <concepts>
#include "hnnet/types.h"

namespace hnnet::builtin {
    template <typename Backend, typename Net, typename Inputs, typename Targets>
        concept BackendType = DatasetType<Inputs> and
                              DatasetType<Targets> and
                              requires(Backend &backend,
                                       Net &net,
                                       const Inputs &inputs,
                                       const Targets &targets) {
                                  { backend.init(net) } -> std::same_as<void>;
                                  { backend.to_device(net, inputs, targets) } -> std::same_as<void>;
                                  { backend.to_host(net) } -> std::same_as<void>;
                              };
}