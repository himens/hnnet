#pragma once
#include <vector>
#include <algorithm>
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
            void receive_signal(const SynapticConn* const conn) {
                const auto in_connections{get_in_connections()}; // static?
                static std::vector<const SynapticConn*> in_connections_left{in_connections};
                const auto found = std::any_of(std::begin(in_connections_left), std::end(in_connections_left), [&] (const auto &in_conn) { return in_conn == conn; });
                if (not found) {
                    throw std::invalid_argument("Neuron::receive_signal: invalid connection!");
                }
                _weighted_sum += conn->get_weighted_signal();
                std::erase(in_connections_left, conn);
                if (in_connections_left.empty()) {
                    _signal = activation(_weighted_sum);
                    broadcast_signal();
                    _weighted_sum = 0.0;
                    in_connections_left = in_connections;
                }
            }
            // Broadcast signal to all receiver neurons
            void broadcast_signal() const {
                for (const auto &conn : get_out_connections()) { // use: for (auto conn : _connections) if (conn->tx == self) ...? 
                    conn->transmit_signal();
                }
            }
            // Add connection to neuron
            void add_connection(SynapticConn* const conn) {
                if (conn == nullptr) {
                    throw std::invalid_argument("Neuron::add_connection: nullptr connection!");
                }
                if (std::any_of(std::begin(_connections), std::end(_connections), [&] (const auto &existing_conn) { return existing_conn == conn; })) {
                    throw std::invalid_argument("Neuron::add_connection: duplicate connection!");
                }
                if (conn->tx != this or conn->rx != this) {
                    throw std::invalid_argument("Neuron::add_connection: invalid connection!");
                }
                _connections.push_back(conn);
            }
            // Get input connections
            std::vector<SynapticConn*> get_in_connections() const {
                return _connections 
                        | std::view::transform([] (const auto &conn) { return conn->rx == this; ) 
                        | std::ranges::to<std::vector>();
            }
            // Get output connections
            std::vector<SynapticConn*> get_out_connections() const {
                return _connections 
                        | std::view::transform([] (const auto &conn) { return conn->tx == this; ) 
                        | std::ranges::to<std::vector>();
            }
            // Learn from data
            virtual bool learn(const real_t data) {
                if (_signal == data) { // use tolerance! it's a real_t comparison!
                    return true;
                }
                bool learnt{false};
                for (auto &conn : get_in_connections()) { // use: for (auto conn : _connections) if (conn->rx == self) ...?
                    update(conn, data);
                    //learnt = conn->tx->learn(data);
                }
                return learnt;
            }
        private:
            // Update conenction weight according to data and a learining rule
            virtual void update(SynapticConn *conn, const real_t data) {
                conn->weight += conn->get_weighted_signal() * data;
            }
            // Activation function
            virtual real_t activation(const real_t signal) const {
                static const real_t threshold{0.2};
                return _weighted_sum >= threshold ? +1 : _weighted_sum < threshold ? -1 : 0;
            }
            // Data members
            real_t _signal{0.0};
            real_t _weighted_sum{0.0};
            std::vector<SynapticConn*> _connections();
    };
}
