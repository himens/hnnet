#pragma once
#include <concepts>
#include "hnnet/types.h"

namespace hnnet {
    template <typename Rule, typename Net, typename Inputs, typename Targets>
        concept LearningRuleType = DatasetType<Inputs> and
                                   DatasetType<Targets> and
                                   requires(Rule& rule,
                                            Net& net,
                                            const Inputs &inputs,
                                            const Targets &targets) {
                                       { rule.init(net) } -> std::same_as<void>;
                                       { rule.learn(net, inputs, targets) } -> std::same_as<bool>;
                                   };
}