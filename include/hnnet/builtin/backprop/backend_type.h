#pragma once
#include <concepts>
#include "hnnet/builtin/backend_type.h"
#include "hnnet/loss_type.h"

namespace hnnet::builtin::backprop {
    template <typename Backend, typename Net, typename Inputs, typename Targets, typename Loss>
        concept BackendType = builtin::BackendType<Backend, Net, Inputs, Targets> and
                               LossType<Loss> and
                               requires(Backend &backend,
                                        Net &net,
                                        index_t begin,
                                        index_t end,
                                        const Loss &loss) {
                                   { backend.forward_backward(net, begin, end, loss) } -> std::same_as<real_t>;
                                   { backend.update_weights() } -> std::same_as<void>;
                                };
}