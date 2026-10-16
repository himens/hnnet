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
        uint bias_count;
        uint batch_begin;
        uint batch_size;
        uint level_begin;
        uint level_count;
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
    device const uint *input_neurons [[buffer(2)]],
    device const uint *bias_neurons [[buffer(3)]],
    device const float *inputs [[buffer(4)]],
    device float *signals [[buffer(5)]],
    device float *weighted_sums [[buffer(6)]],
    device float *deltas [[buffer(7)]],
    uint sample [[thread_position_in_grid]]) {
    if (sample >= parameters.batch_size) return;
    const uint offset = sample * parameters.neuron_count;
    for (uint neuron = 0; neuron < parameters.neuron_count; ++neuron) {
        signals[offset + neuron] = 0.0f;
        weighted_sums[offset + neuron] = 0.0f;
        deltas[offset + neuron] = 0.0f;
    }
    for (uint input = 0; input < parameters.input_count; ++input) {
        const uint neuron = input_neurons[input];
        const float value = inputs[(parameters.batch_begin + sample) * parameters.input_count + input];
        weighted_sums[offset + neuron] = value;
        signals[offset + neuron] = activate(activations[neuron], value);
    }
    for (uint bias = 0; bias < parameters.bias_count; ++bias) {
        const uint neuron = bias_neurons[bias];
        weighted_sums[offset + neuron] = 1.0f;
        signals[offset + neuron] = activate(activations[neuron], 1.0f);
    }
}

kernel void hnnet_forward(
    constant Parameters &parameters [[buffer(0)]],
    device const ActivationDescriptor *activations [[buffer(1)]],
    device const uint *level_neurons [[buffer(2)]],
    device const uint *incoming_offsets [[buffer(3)]],
    device const uint *incoming_edges [[buffer(4)]],
    device const uint *sources [[buffer(5)]],
    device const float *weights [[buffer(6)]],
    device float *signals [[buffer(7)]],
    device float *weighted_sums [[buffer(8)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * parameters.level_count) return;
    const uint offset = (item / parameters.level_count) * parameters.neuron_count;
    const uint neuron = level_neurons[parameters.level_begin + item % parameters.level_count];
    float sum = 0.0f;
    for (uint entry = incoming_offsets[neuron]; entry < incoming_offsets[neuron + 1]; ++entry) {
        const uint edge = incoming_edges[entry];
        sum += weights[edge] * signals[offset + sources[edge]];
    }
    weighted_sums[offset + neuron] = sum;
    signals[offset + neuron] = activate(activations[neuron], sum);
}

kernel void hnnet_backward(
    constant Parameters &parameters [[buffer(0)]],
    device const ActivationDescriptor *activations [[buffer(1)]],
    device const uint *level_neurons [[buffer(2)]],
    device const uint *outgoing_offsets [[buffer(3)]],
    device const uint *outgoing_edges [[buffer(4)]],
    device const uint *receivers [[buffer(5)]],
    device const float *weights [[buffer(6)]],
    device const uint *output_positions [[buffer(7)]],
    device const float *targets [[buffer(8)]],
    device const float *signals [[buffer(9)]],
    device const float *weighted_sums [[buffer(10)]],
    device float *deltas [[buffer(11)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= parameters.batch_size * parameters.level_count) return;
    const uint sample = item / parameters.level_count;
    const uint offset = sample * parameters.neuron_count;
    const uint neuron = level_neurons[parameters.level_begin + item % parameters.level_count];
    float delta = 0.0f;
    for (uint entry = outgoing_offsets[neuron]; entry < outgoing_offsets[neuron + 1]; ++entry) {
        const uint edge = outgoing_edges[entry];
        delta += weights[edge] * deltas[offset + receivers[edge]];
    }
    const float slope = derivative(activations[neuron], weighted_sums[offset + neuron]);
    const uint output = output_positions[neuron];
    if (output != 0xffffffffu) {
        const float target = targets[(parameters.batch_begin + sample) * parameters.output_count + output];
        delta += (signals[offset + neuron] - target) * slope;
    }
    else {
        delta *= slope;
    }
    deltas[offset + neuron] = delta;
}

kernel void hnnet_loss(
    constant Parameters &parameters [[buffer(0)]],
    device const uint *output_neurons [[buffer(1)]],
    device const float *targets [[buffer(2)]],
    device const float *signals [[buffer(3)]],
    device float *losses [[buffer(4)]],
    uint sample [[thread_position_in_grid]]) {
    if (sample >= parameters.batch_size) return;
    float loss = 0.0f;
    for (uint output = 0; output < parameters.output_count; ++output) {
        const float target = targets[(parameters.batch_begin + sample) * parameters.output_count + output];
        const float signal = signals[sample * parameters.neuron_count + output_neurons[output]];
        const float error = target - signal;
        loss += 0.5f * error * error;
    }
    losses[sample] = loss;
}

kernel void hnnet_gradient(
    constant Parameters &parameters [[buffer(0)]],
    device const uint *sources [[buffer(1)]],
    device const uint *receivers [[buffer(2)]],
    device const float *signals [[buffer(3)]],
    device const float *deltas [[buffer(4)]],
    device float *gradient [[buffer(5)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= parameters.connection_count) return;
    float sum = 0.0f;
    for (uint sample = 0; sample < parameters.batch_size; ++sample) {
        const uint offset = sample * parameters.neuron_count;
        sum += deltas[offset + receivers[edge]] * signals[offset + sources[edge]];
    }
    gradient[edge] = sum / float(parameters.batch_size);
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