#pragma once
#include "hnnet/types.h"

namespace hNNet {
    template <typename Rule, typename Net, typename Inputs, typename Targets>
        concept LearningRuleType =
            DatasetType<Inputs> and DatasetType<Targets> and
            requires (Rule& rule, Net& net, const Inputs &inputs, const Targets &targets) {
                { rule.learn(net, inputs, targets) } -> std::same_as<real_t>;
            };
}