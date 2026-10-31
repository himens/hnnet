#pragma once
#include <concepts>
#include <span>
#include "hnnet/backend_type.h"
#include "hnnet/loss_type.h"
#include "hnnet/types.h"

namespace hnnet::backprop {
    template <DatasetType Inputs, DatasetType Targets>
        struct TrainingData {
            const Inputs &inputs;
            const Targets &targets;
        };
    template <typename Backend, typename View, typename Inputs, typename Targets, typename Loss>
        concept BackendType = DatasetType<Inputs> and
                              DatasetType<Targets> and
                              hnnet::BackendType<Backend, View, TrainingData<Inputs, Targets>, weight_vector_t> and
                              LossType<Loss> and
                              requires(Backend &backend,
                                       View &view,
                                       index_t begin,
                                       index_t end,
                                       const Loss &loss) {
                                  { backend.backprop(begin, end, loss) } -> std::same_as<real_t>;
                              };
}
