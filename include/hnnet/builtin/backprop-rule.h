#pragma once
#include "hnnet/nnet.h"

namespace hNNet::Builtin {
    class BackpropRule {
        public:
            // Constructor
            explicit BackpropRule(const real_t learning_rate, const real_t momentum = 0.0) : _learning_rate(learning_rate), _momentum(momentum) {}
            // Learn from targets using the back-propagation learning rule
            template <NNetType Net>
                real_t learn(Net &net, const output_t<Net> &targets) {
                    auto view = net.view();
                    _deltas.assign(view.neuron_count(), 0.0);
                    if (_dweights.empty()) {
                        _dweights.resize(view.connection_count(), 0.0);
                    }
                    // seed output deltas with the error already scaled by the activation derivative
                    real_t squared_error{0.0};
                    for (const auto &[target, iout] : std::views::zip(targets, view.iout_neurons())) {
                        const auto &out_neuron = view.neuron(iout);
                        const auto error = (target - out_neuron.signal());
                        squared_error += error * error;
                        _deltas[iout] = error * out_neuron.activation()->derivative(out_neuron.weighted_sum());
                    }
                    // partitions are already in topological order: walking them backwards guarantees that,
                    // by the time a partition's irx is reached, all its downstream contributions are collected
                    for (const auto &partition : view.partitions() | std::views::reverse) {
                        const auto &rx = view.neuron(partition.irx);
                        if (rx.type() != NeuronType::output) {
                            _deltas[partition.irx] *= rx.activation()->derivative(rx.weighted_sum());
                        }
                        for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                            const auto itx = view.itx(icon);
                            _deltas[itx] += _deltas[partition.irx] * view.weight(icon);
                        }
                    }
                    // connections in a partition share their receiver and therefore its delta
                    for (const auto &partition : view.partitions()) {
                        const auto scaled_delta = _learning_rate * _deltas[partition.irx];
                        #pragma omp simd
                        for (size_t icon = partition.begin; icon < partition.end; ++icon) {
                            const auto change = scaled_delta * view.neuron(view.itx(icon)).signal()
                                              + _momentum * _dweights[icon];
                            view.weight(icon) += change;
                            _dweights[icon] = change;
                        }
                    }
                    return squared_error;
                }
        private:
            // Data members
            real_t _learning_rate{0.0};
            real_t _momentum{0.0};
            std::vector<real_t> _deltas{};
            std::vector<real_t> _dweights{};
    };
}

