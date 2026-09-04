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
                        const auto error = (target - view.signal(iout));
                        squared_error += error * error;
                        _deltas[iout] = error * out_neuron.activation()->derivative(out_neuron.weighted_sum());
                    }
                    // partitions are already in topological order: walk them backwards
                    const auto partitions = view.partitions();
                    auto reversed_partitions = partitions | std::views::reverse;
                    for (index_t ipart{0}; ipart < reversed_partitions.size(); ipart++) {
                        const auto &partition = reversed_partitions[ipart];
                        const auto block_id = partition.block_id;
                        if (block_id != Net::no_block) {
                            const auto &block = view.dense_block(block_id);
                            for (index_t irow{0}; irow < block.rx_count; ++irow) {
                                const auto irx = block.rx_begin + irow;
                                const auto &rx = view.neuron(irx);
                                auto &delta_rx = _deltas[irx];
                                if (rx.type() != NeuronType::output) {
                                    delta_rx *= rx.activation()->derivative(rx.weighted_sum());
                                }
                                //#pragma omp simd
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                for (index_t icol = 0; icol < block.tx_count; ++icol) {
                                    _deltas[block.tx_begin + icol] += delta_rx * view.weight(row_offset + icol);
                                }
                            }
                            ipart += block.rx_count - 1;
                            continue;
                        }
                        const auto &rx = view.neuron(partition.irx);
                        auto &delta_rx = _deltas[partition.irx];
                        if (rx.type() != NeuronType::output) {
                            delta_rx *= rx.activation()->derivative(rx.weighted_sum());
                        }
                        for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                            const auto itx = view.connection(icon).itx;
                            _deltas[itx] += delta_rx * view.weight(icon);
                        }
                    }
                    // update weights
                    for (index_t ipart{0}; ipart < partitions.size(); ipart++) {
                        const auto &partition = partitions[ipart];
                        const auto block_id = partition.block_id;
                        if (block_id != Net::no_block) {
                            const auto &block = view.dense_block(block_id);
                            for (index_t irow{0}; irow < block.rx_count; ++irow) {
                                const auto scaled_delta = _learning_rate * _deltas[block.rx_begin + irow];
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                //#pragma omp simd
                                for (index_t icol = 0; icol < block.tx_count; ++icol) {
                                    const auto dweight = scaled_delta * view.signal(block.tx_begin + icol) + (_momentum * _dweights[row_offset + icol]);
                                    view.weight(row_offset + icol) += dweight;
                                    _dweights[row_offset + icol] = dweight;
                                }
                            }
                            ipart += block.rx_count - 1;
                            continue;
                        }
                        for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                            const auto irx = view.connection(icon).irx; // indirection
                            const auto itx = view.connection(icon).itx; // indirection
                            const auto dweight = (_learning_rate * _deltas[irx] * view.signal(itx)) + (_momentum * _dweights[icon]);
                            view.weight(icon) += dweight;
                            _dweights[icon] = dweight;
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
