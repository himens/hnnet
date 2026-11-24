#ifndef HNNET_METAL_UPDATE_INCLUDED
#define HNNET_METAL_UPDATE_INCLUDED
#include <metal_stdlib>

namespace hnnet {
namespace dense {
namespace update {
namespace metal {
    struct Parameters {
        uint neuron_count;
        uint connection_count;
        uint input_count;
        uint output_count;
        uint output_begin;
        uint batch_begin;
        uint batch_size;
    };

    struct DenseLayer {
        uint tx_begin;
        uint tx_count;
        uint rx_begin;
        uint rx_count;
        uint weight_offset;
    };
}
}
}
}

kernel void hnnet_seed(
    constant hnnet::dense::update::metal::Parameters &parameters [[buffer(0)]],
    device const hnnet::Activation *activations [[buffer(1)]],
    device const uint *input_positions [[buffer(2)]],
    device const float *inputs [[buffer(3)]],
    device float *signals [[buffer(4)]],
    device float *weighted_sums [[buffer(5)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * parameters.neuron_count) return;
    const uint neuron = item % parameters.neuron_count;
    const uint input = input_positions[neuron];
    float signal = 0.0f;
    float weighted_sum = 0.0f;
    if (input < parameters.input_count) {
        const uint sample = parameters.batch_begin + item / parameters.neuron_count;
        weighted_sum = inputs[sample * parameters.input_count + input];
        const auto activation = activations[neuron];
        signal = activation(weighted_sum);
    }
    else if (input == 0xfffffffeu) {
        weighted_sum = 1.0f;
        const auto activation = activations[neuron];
        signal = activation(weighted_sum);
    }
    signals[item] = signal;
    weighted_sums[item] = weighted_sum;
}

kernel void hnnet_dense_activation(
    constant hnnet::dense::update::metal::Parameters &parameters [[buffer(0)]],
    constant hnnet::dense::update::metal::DenseLayer &layer [[buffer(1)]],
    device const hnnet::Activation *activations [[buffer(2)]],
    device const float *weighted_sums [[buffer(3)]],
    device float *signals [[buffer(4)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * layer.rx_count) return;
    const uint row = item % layer.rx_count;
    const uint neuron = layer.rx_begin + row;
    const uint sample_offset = (item / layer.rx_count) * parameters.neuron_count;
    const uint offset = sample_offset + neuron;
    const auto activation = activations[neuron];
    signals[offset] = activation(weighted_sums[offset]);
}
#endif
