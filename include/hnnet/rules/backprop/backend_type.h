#pragma once
#include <concepts>
#include <span>
#include "hnnet/backend_type.h"
#include "hnnet/loss_type.h"
#include "hnnet/types.h"

namespace hnnet::backprop {
    template <typename Backend, typename Data, typename View, typename Loss>
        concept BackendType = hnnet::BackendType<Backend, Data, std::vector<real_t>, View> and
                              TrainingDataType<Data> and
                              LossType<Loss> and
                              requires(Backend &backend,
                                       index_t begin,
                                       index_t end,
                                       const Loss &loss) {
                                  { backend.backprop(begin, end, loss) } -> std::same_as<real_t>;
                              };
}
