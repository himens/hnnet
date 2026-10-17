#include <metal_stdlib>
#ifndef HNNET_METAL_ACTIVATIONS_INCLUDED
#include "activations.h"
#endif

namespace hnnet {
namespace backprop {
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

    struct ActivationDescriptor {
        uint kind;
        float sigma;
    };

    struct UpdateParameters {
        uint weight_count;
        float first;
        float second;
    };

    struct DenseLayer {
        uint tx_begin;
        uint tx_count;
        uint rx_begin;
        uint rx_count;
        uint weight_offset;
        uint weight_stride;
        uint bias_neuron;
    };

    inline float activate(const ActivationDescriptor descriptor, const float value) {
        switch (descriptor.kind) {
            case 0: return activation::identity(value);
            case 1: return activation::sigmoid(value, descriptor.sigma);
            case 2: return activation::relu(value);
        }
        return 0.0f;
    }

    inline float derivative(const ActivationDescriptor descriptor, const float value) {
        switch (descriptor.kind) {
            case 0: return activation::identity_derivative(value);
            case 1: return activation::sigmoid_derivative(value, descriptor.sigma);
            case 2: return activation::relu_derivative(value);
        }
        return 0.0f;
    }
}
}
}

using namespace hnnet::backprop::metal;

kernel void hnnet_seed(
    constant Parameters &parameters [[buffer(0)]],
    device const ActivationDescriptor *activations [[buffer(1)]],
    device const uint *input_positions [[buffer(2)]],
    device const float *inputs [[buffer(3)]],
    device float *signals [[buffer(4)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * parameters.neuron_count) return;
    const uint neuron = item % parameters.neuron_count;
    const uint input = input_positions[neuron];
    float signal = 0.0f;
    if (input < parameters.input_count) {
        const uint sample = parameters.batch_begin + item / parameters.neuron_count;
        signal = activate(activations[neuron], inputs[sample * parameters.input_count + input]);
    }
    else if (input == 0xfffffffeu) {
        signal = activate(activations[neuron], 1.0f);
    }
    signals[item] = signal;
}

kernel void hnnet_loss(
    constant Parameters &parameters [[buffer(0)]],
    device const float *targets [[buffer(1)]],
    device const float *signals [[buffer(2)]],
    device float *losses [[buffer(3)]],
    uint sample [[thread_position_in_grid]]) {
    if (sample >= parameters.batch_size) return;
    float loss = 0.0f;
    for (uint output = 0; output < parameters.output_count; ++output) {
        const float target = targets[(parameters.batch_begin + sample) * parameters.output_count + output];
        const float signal = signals[sample * parameters.neuron_count + parameters.output_begin + output];
        const float error = target - signal;
        loss += 0.5f * error * error;
    }
    losses[sample] = loss;
}

kernel void hnnet_dense_activation(
    constant Parameters &parameters [[buffer(0)]],
    constant DenseLayer &layer [[buffer(1)]],
    device const ActivationDescriptor *activations [[buffer(2)]],
    device const float *weights [[buffer(3)]],
    device float *weighted_sums [[buffer(4)]],
    device float *signals [[buffer(5)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * layer.rx_count) return;
    const uint row = item % layer.rx_count;
    const uint neuron = layer.rx_begin + row;
    const uint sample_offset = (item / layer.rx_count) * parameters.neuron_count;
    const uint offset = sample_offset + neuron;
    float sum = weighted_sums[offset];
    if (layer.bias_neuron != 0xffffffffu) {
        sum += weights[layer.weight_offset + row * layer.weight_stride + layer.tx_count]
            * signals[sample_offset + layer.bias_neuron];
    }
    weighted_sums[offset] = sum;
    signals[offset] = activate(activations[neuron], sum);
}

kernel void hnnet_dense_delta(
    constant Parameters &parameters [[buffer(0)]],
    constant DenseLayer &layer [[buffer(1)]],
    device const ActivationDescriptor *activations [[buffer(2)]],
    device const float *targets [[buffer(3)]],
    device const float *signals [[buffer(4)]],
    device const float *weighted_sums [[buffer(5)]],
    device float *deltas [[buffer(6)]],
    device const float *propagated_deltas [[buffer(7)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * layer.rx_count) return;
    const uint sample = item / layer.rx_count;
    const uint row = item % layer.rx_count;
    const uint neuron = layer.rx_begin + row;
    const uint offset = sample * parameters.neuron_count + neuron;
    const float slope = derivative(activations[neuron], weighted_sums[offset]);
    if (layer.rx_begin == parameters.output_begin) {
        const float target = targets[(parameters.batch_begin + sample) * parameters.output_count + row];
        deltas[offset] = (signals[offset] - target) * slope;
    }
    else {
        deltas[offset] = propagated_deltas[offset] * slope;
    }
}

kernel void hnnet_bias_gradient(
    constant Parameters &parameters [[buffer(0)]],
    constant DenseLayer &layer [[buffer(1)]],
    device const float *signals [[buffer(2)]],
    device const float *deltas [[buffer(3)]],
    device float *gradient [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= layer.rx_count) return;
    float sum = 0.0f;
    for (uint sample = 0; sample < parameters.batch_size; ++sample) {
        const uint offset = sample * parameters.neuron_count;
        sum += deltas[offset + layer.rx_begin + row] * signals[offset + layer.bias_neuron];
    }
    gradient[layer.weight_offset + row * layer.weight_stride + layer.tx_count] = sum / float(parameters.batch_size);
}

kernel void hnnet_momentum(
    device float *weights [[buffer(0)]],
    device const float *gradient [[buffer(1)]],
    device float *previous_delta [[buffer(2)]],
    constant UpdateParameters &parameters [[buffer(3)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= parameters.weight_count) return;
    const float delta = -parameters.first * gradient[edge] + parameters.second * previous_delta[edge];
    weights[edge] += delta;
    previous_delta[edge] = delta;
}

kernel void hnnet_rprop(
    device float *weights [[buffer(0)]],
    device const float *gradient [[buffer(1)]],
    device float *delta [[buffer(2)]],
    device float *previous_gradient [[buffer(3)]],
    constant UpdateParameters &parameters [[buffer(4)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= parameters.weight_count) return;
    const float grad = gradient[edge];
    const float product = previous_gradient[edge] * grad;
    if (product > 0.0f) {
        delta[edge] = ::metal::min(parameters.first * delta[edge], 50.0f);
    }
    else if (product < 0.0f) {
        delta[edge] = ::metal::max(parameters.second * delta[edge], 1e-6f);
        previous_gradient[edge] = 0.0f;
        return;
    }
    const float direction = float(grad > 0.0f) - float(grad < 0.0f);
    weights[edge] -= direction * delta[edge];
    previous_gradient[edge] = grad;
}