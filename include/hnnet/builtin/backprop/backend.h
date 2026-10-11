#pragma once
#include "hnnet/loss.h"
#include "hnnet/builtin/backprop/optimizer.h"

namespace hnnet::builtin {
    // Backprop backend concept
    template <typename Backend, typename Net, typename Inputs, typename Targets, typename Optimizer, typename Loss>
        concept BackpropBackendType = DatasetType<Inputs> and 
                                      DatasetType<Targets> and 
                                      OptimizerType<Optimizer> and
                                      LossType<Loss> and
                                      requires(Backend &backend, 
                                               Net &net, 
                                               const Inputs &inputs, 
                                               const Targets &targets,
                                               index_t begin, 
                                               index_t end, 
                                               Optimizer &optimizer, 
                                               const Loss &loss) {
                                          { backend.to_device(net, inputs, targets) } -> std::same_as<void>;
                                          { backend.forward_backward(net, begin, end, loss) } -> std::same_as<real_t>;
                                          { backend.update_weights(net, optimizer) } -> std::same_as<void>;
                                          { backend.to_host(net) } -> std::same_as<void>;
                                      };
}