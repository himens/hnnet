#pragma once
#include "hnnet.h"

namespace hNNet {
    //////////////////
    // Neuron class //
    //////////////////
    class Neuron {
        public:
            struct SynapticConn {
                // Get weighted signal
                real_t get_weighted_signal() const {
                    if (tx == nullptr) {
                        throw std::invalid_argument("SynapticConn::get_weighted_signal: nullptr tx!");
                    }
                    return tx->get_signal() * weight;
                }
                // Transmit signal
                void transmit_signal() const {
                    if (rx == nullptr) {
                        throw std::invalid_argument("SynapticConn::transmit_signal: nullptr rx!");
                    }
                    rx->receive_signal(this);
                }
                // Data members
                const Neuron* const tx{nullptr};
                Neuron* const rx{nullptr};
                real_t weight{0.0};
            };
        public:
            // Get neuron signal 
            real_t get_signal() const {
                return _signal;
            } 
            // Set neuron signal
            void set_signal(const real_t signal) {
                _signal = signal;
            }
            // Receive signal from synaptic connection
            void receive_signal(const SynapticConn* conn) {
                static real_t weighted_sum{0.0};
                static std::vector<SynapticConn*> in_connections_left{get_in_connections()};
                const auto found = std::any_of(std::begin(in_connections_left), std::end(in_connections_left), [&] (const auto &in_conn) { return in_conn == conn; });
                if (not found) {
                    throw std::invalid_argument("Neuron::receive_signal: invalid connection!");
                }
                weighted_sum += conn->get_weighted_signal();
                std::erase(in_connections_left, conn);
                //std::println("Neuron::receive_signal: signal: {}, weight: {}, left: {}", conn->get_weighted_signal(), conn->weight, in_connections_left.size());
                if (in_connections_left.empty()) {
                    _signal = activation(weighted_sum);
                    //std::println("Neuron::receive_signal: all signals received! Signal: {}, weighted sum: {}", _signal, weighted_sum);
                    broadcast_signal();
                    weighted_sum = 0.0;
                    in_connections_left = get_in_connections();
                }
            }
            // Broadcast signal to all receiver neurons
            void broadcast_signal() const {
                for (const auto &conn : get_out_connections()) {
                    conn->transmit_signal();
                }
            }
            // Add connection to neuron
            void add_connection(SynapticConn* conn) {
                if (conn == nullptr) {
                    throw std::invalid_argument("Neuron::add_connection: nullptr connection!");
                }
                const auto found = std::any_of(std::begin(_connections), std::end(_connections),
                        [&] (const auto &existing_conn) { return existing_conn == conn; });
                if (found) {
                    throw std::invalid_argument("Neuron::add_connection: duplicate connection!");
                }
                if ((conn->tx != this) and (conn->rx != this)) {
                    throw std::invalid_argument("Neuron::add_connection: invalid connection!");
                }
                _connections.push_back(conn);
            }
            // Get input connections
            std::vector<SynapticConn*> get_in_connections() const {
                return _connections 
                        | std::views::filter([&] (const auto &conn) { return conn->rx == this; } ) 
                        | std::ranges::to<std::vector>();
            }
            // Get output connections
            std::vector<SynapticConn*> get_out_connections() const {
                return _connections 
                        | std::views::filter([&] (const auto &conn) { return conn->tx == this; } ) 
                        | std::ranges::to<std::vector>();
            }
            // Learn from target data
            virtual bool learn(const real_t target) {
                const auto learnt = std::abs(_signal - target) < 1e-6;
                if (not learnt) {
                    for (auto &conn : get_in_connections()) {
                        update(conn, target);
                        std::println("Neuron::learn: updated weight of connection {}: {}", static_cast<void*>(conn), conn->weight);
                    }
                }
                return learnt;
            }
        private:
            // Update connection weight according to target and a learning rule
            virtual void update(SynapticConn* conn, const real_t target) = 0;
            // Activation function
            virtual real_t activation(const real_t weighted_sum) const = 0;
            // Data members
            real_t _signal{0.0};
            std::vector<SynapticConn*> _connections{};
    };

    template <typename T>
        concept NeuronType = std::derived_from<T, Neuron>;
    template <typename T>
        concept NeuronPtr = std::is_pointer_v<std::remove_cvref_t<T>> and NeuronType<std::remove_pointer_t<std::remove_cvref_t<T>>>;
    template <typename T>
        concept NeuronRange = std::ranges::range<T> and NeuronPtr<std::ranges::range_value_t<T>>;
}