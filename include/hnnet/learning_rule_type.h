#pragma once
#include <concepts>
#include <vector>
#include "hnnet/types.h"

namespace hnnet {
    template <typename Rule, typename View, typename Data>
        concept LearningRuleType = TrainingDataType<Data> and
                                   requires(Rule& rule,
                                            View& view,
                                            const Data &data,
                                            std::vector<real_t> &weights) {
                                       { rule.init(view, data) } -> std::same_as<void>;
                                       { rule.learn() } -> std::same_as<bool>;
                                       { rule.apply(weights) } -> std::same_as<void>;
                                   };
}