#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class BackpropRule {
        public:
            // Constructor
            explicit BackpropRule(const real_t learning_rate, const real_t momentum = 0.0) : _learning_rate(learning_rate), _momentum(momentum) {}
            // Learn from targets using the back-propagation learning rule
            template <NNetType Net>
                void learn(Net &net, const output_t<Net> &targets) {
                    auto view = net.view();
                    _deltas.assign(view.neuron_count(), 0.0);
                    if (_dweights.empty()) {
                        _dweights.resize(view.connection_count(), 0.0);
                    }
                    // Seed output deltas with the error already scaled by the activation derivative
                    index_t itarget{0};
                    for (size_t ineuron{0}; ineuron < view.neuron_count(); ++ineuron) {
                        const auto &neuron = view.neuron(ineuron);
                        if (neuron.type() != NeuronType::output) {
                            continue;  // Skip non-output neurons
                        }
                        const auto error = (targets[itarget++] - neuron.signal());
                        _deltas[ineuron] = error * neuron.activation()->derivative(neuron.weighted_sum());
                    }
                    // Partitions are already in topological order: walking them backwards guarantees that,
                    // by the time a partition's irx is reached, all its downstream contributions are collected
                    for (const auto &partition : view.partitions() | std::views::reverse) {
                        const auto &rx = view.neuron(partition.irx);
                        if (rx.type() != NeuronType::output) {
                            _deltas[partition.irx] *= rx.activation()->derivative(rx.weighted_sum());
                        }
                        for (const auto &iconn : std::views::iota(partition.begin, partition.end)) {
                            const auto itx = view.itx(iconn);
                            _deltas[itx] += _deltas[partition.irx] * view.weight(iconn);
                        }
                    }
                    // Update weights for all connections
                    for (index_t iconn{0}; iconn < view.connection_count(); ++iconn) {
                        const auto irx = view.irx(iconn);
                        const auto &tx = view.neuron(view.itx(iconn));
                        auto &weight = view.weight(iconn);
                        const auto old_weight = weight;
                        weight += (_learning_rate * _deltas[irx] * tx.signal()) + (_momentum * _dweights[iconn]);
                        _dweights[iconn] = (weight - old_weight);  // Store the weight change for momentum
                    }
                }
        private:
            // Data members
            real_t _learning_rate{0.0};
            real_t _momentum{0.0};
            std::vector<real_t> _deltas{};
            std::vector<real_t> _dweights{};
    };
}

