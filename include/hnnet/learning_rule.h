#pragma once
#include "hnnet/types.h"

namespace hNNet {
    // A LearningRule takes over an entire epoch: it receives all training samples and
    // is free to decide how to iterate them (online, mini-batch, parallel, ...).
    template <typename Rule, typename Net>
        concept LearningRuleType = 
            requires { typename Net::output_type; typename Net::TrainingData; } and 
            requires (Rule& rule, Net& net, std::span<const typename Net::TrainingData> samples) {
                { rule.learn(net, samples) } -> std::same_as<real_t>;
            };
}

