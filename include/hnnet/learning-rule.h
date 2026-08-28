#pragma once
#include "hnnet/types.h"

namespace hNNet {
    template <typename Rule, typename Net>
        concept LearningRuleType = 
            requires { typename Net::output_type; } and 
            requires (Rule& rule, Net& net, const typename Net::output_type& targets) { 
                { rule.learn(net, targets) } -> std::same_as<real_t>;
            };
}
