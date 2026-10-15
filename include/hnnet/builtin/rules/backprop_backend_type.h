#pragma once
#include <concepts>
#include "hnnet/backend_type.h"
#include "hnnet/loss_type.h"

namespace hnnet::builtin::rule {
    template <typename Backend, typename Net, typename Inputs, typename Targets, typename Loss>
        concept BackpropBackendType = hnnet::BackendType<Backend, Net, Inputs, Targets> and
                                      LossType<Loss> and
                                      requires(Backend &backend,
                                               Net &net,
                                               index_t begin,
                                               index_t end,
                                               const Loss &loss) {
                                          { backend.backprop(net, begin, end, loss) } -> std::same_as<real_t>;
                                      };
}
