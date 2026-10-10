#pragma once
#include "hnnet/builtin/backprop/optimizers.h"
#include "hnnet/builtin/backprop/omp_backend.h"

namespace hnnet::builtin {
    ////////////////////////
    // BackpropRule class //
    ////////////////////////
    template <typename Optimizer, LossType Loss = MSELoss, typename Backend = OpenMPBackend>
        class BackpropRule {
            public:
                // Constructor
                explicit BackpropRule(Optimizer optimizer = Optimizer{}, Backend backend = {}, const int_t batch_size = 1, Loss loss = {}) : _optimizer(optimizer), _backend(backend), _batch_size(batch_size), _loss(loss) {
                    if (batch_size == 0) {
                        throw std::invalid_argument("BackpropRule::BackpropRule: batch_size must not be zero");
                    }
                }
                // Learn from a whole epoch of training samples
                template <typename Net, DatasetType Inputs, DatasetType Targets>
                    requires BackpropBackendType<Backend, Net, Inputs, Targets, Optimizer, Loss>
                    real_t learn(Net &net, const Inputs &inputs, const Targets &targets) {
                        const auto sample_count = static_cast<index_t>(std::ranges::size(inputs));
                        if (sample_count == 0) {
                            throw std::invalid_argument("BackpropRule::learn: inputs must not be empty");
                        }
                        const auto batch_size = _batch_size < 0 ? sample_count : std::min(_batch_size, sample_count);
                        const auto batch_count = utils::math::ceil(sample_count, batch_size);
                        _backend.to_device(net, inputs, targets);
                        real_t loss{0.0_real};
                        for (auto ibatch{0}; ibatch < batch_count; ++ibatch) {
                            const auto batch_begin = ibatch * batch_size;
                            const auto batch_end = std::min(sample_count, batch_begin + batch_size);
                            loss += _backend.forward_backward(net, inputs, targets, batch_begin, batch_end, _loss);
                            _backend.update_weights(net, _optimizer);
                        }
                        _backend.to_host(net);
                        return loss / sample_count;
                    }
            private:
                Optimizer _optimizer;
                int_t _batch_size{1};
                Loss _loss;
                Backend _backend;
        };
}