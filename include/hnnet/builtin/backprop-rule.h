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
                    const auto &partitions = view.partitions();
                    size_t ipart{partitions.size()};
                    while (ipart > 0) {
                        --ipart;
                        const auto block_id = view.block_id(ipart);
                        if (block_id != view.no_dense_block()) {
                            const auto &block = view.dense_blocks()[block_id];
                            for (index_t irow{0}; irow < block.rx_count; ++irow) {
                                const auto irx = block.rx_begin + irow;
                                const auto &rx = view.neuron(irx);
                                auto &delta_rx = _deltas[irx];
                                if (rx.type() != NeuronType::output) {
                                    delta_rx *= rx.activation()->derivative(rx.weighted_sum());
                                }
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                // row-major sweep, same access pattern as broadcast(): equivalent to a W^T * delta product without transposing storage
                                #pragma omp simd
                                for (index_t icol = 0; icol < block.tx_count; ++icol) {
                                    _deltas[block.tx_begin + icol] += delta_rx * view.weight(row_offset + icol);
                                }
                            }
                            ipart -= (block.rx_count - 1);  // dense block members are contiguous: skip them all at once
                            continue;
                        }
                        const auto &partition = partitions[ipart];
                        const auto &rx = view.neuron(partition.irx);
                        auto &delta_rx = _deltas[partition.irx];
                        if (rx.type() != NeuronType::output) {
                            delta_rx *= rx.activation()->derivative(rx.weighted_sum());
                        }
                        for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                            const auto itx = view.itx(icon);
                            _deltas[itx] += delta_rx * view.weight(icon);
                        }
                    }
                    // update weights: dense blocks as a rank-1 (outer product) update, the rest via the generic edge path
                    ipart = 0;
                    while (ipart < partitions.size()) {
                        const auto block_id = view.block_id(ipart);
                        if (block_id != view.no_dense_block()) {
                            const auto &block = view.dense_blocks()[block_id];
                            for (index_t irow{0}; irow < block.rx_count; ++irow) {
                                const auto scaled_delta = _learning_rate * _deltas[block.rx_begin + irow];
                                const auto row_offset = block.weight_offset + irow * block.tx_count;
                                #pragma omp simd
                                for (index_t icol = 0; icol < block.tx_count; ++icol) {
                                    const auto dweight = (scaled_delta * view.neuron(block.tx_begin + icol).signal()) + (_momentum * _dweights[row_offset + icol]);
                                    view.weight(row_offset + icol) += dweight;
                                    _dweights[row_offset + icol] = dweight;
                                }
                            }
                            ipart += block.rx_count;  // dense block members are contiguous: skip them all at once
                            continue;
                        }
                        const auto &partition = partitions[ipart];
                        for (const auto &icon : std::views::iota(partition.begin, partition.end)) {
                            const auto irx = view.irx(icon); // indirection
                            const auto &tx = view.neuron(view.itx(icon)); // indirection
                            const auto dweight = (_learning_rate * _deltas[irx] * tx.signal()) + (_momentum * _dweights[icon]);
                            view.weight(icon) += dweight;
                            _dweights[icon] = dweight;
                        }
                        ++ipart;
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
