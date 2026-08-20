#pragma once
#include "hnnet/types.h"

namespace hNNet {
    template <typename Rule, typename Net>
    concept LearningRuleType = requires(Rule& rule, Net& net, const typename Net::OutputData& targets) {
        { rule.learn(net, targets) } -> std::same_as<void>;
    };
}
