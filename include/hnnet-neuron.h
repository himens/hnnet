#pragma once
#include "hnnet-activation.h"

namespace hNNet {
    //////////////////
    // Neuron class //
    //////////////////
    class Neuron {
        public:
            /////////////////////////
            // Synaptic connection //
            /////////////////////////
            struct SynapticConn {
                // Get transmitted signal
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
            // Get weighted sum of input signals
            real_t get_weighted_sum() const {
                return _weighted_sum;
            }
            // Broadcast signal to all receiver neurons
            void broadcast_signal() const {
                for (const auto &conn : get_out_connections()) {
                    conn->transmit_signal();
                }
            }
            // Add connection to neuron
            void add_connection(SynapticConn* new_conn) {
                if (new_conn == nullptr) {
                    throw std::invalid_argument("Neuron::add_connection: nullptr connection!");
                }
                const auto found = std::any_of(std::begin(_in_connections),  std::end(_in_connections),  [&] (const auto &conn) { return conn == new_conn; }) or
                                   std::any_of(std::begin(_out_connections), std::end(_out_connections), [&] (const auto &conn) { return conn == new_conn; });
                if (found) {
                    throw std::invalid_argument("Neuron::add_connection: duplicate connection!");
                }
                if ((new_conn->tx != this) and (new_conn->rx != this)) {
                    throw std::invalid_argument("Neuron::add_connection: invalid connection!");
                }
                if (new_conn->tx == this) {
                    _out_connections.push_back(new_conn);
                }
                else if (new_conn->rx == this) {
                    _in_connections.push_back(new_conn);
                }
            }
            // Get input connections
            std::vector<SynapticConn*> get_in_connections() const {
                return _in_connections;
            }
            // Get output connections
            std::vector<SynapticConn*> get_out_connections() const {
                return _out_connections;
            }
            // Learn from target data
            virtual bool learn(const real_t target) {
                const auto learnt = std::abs(_signal - target) < 1e-6;
                if (not learnt) {
                    for (auto &conn : get_in_connections()) {
                        update(conn, target);
                        //std::println("Neuron::learn: updated weight of connection {}: {}", static_cast<void*>(conn), conn->weight);
                    }
                }
                return learnt;
            }
        protected:
            // Constructor
            Neuron(std::unique_ptr<Activation> activation) {
                if (activation == nullptr) {
                    throw std::invalid_argument("Neuron::Neuron: nullptr activation function!");
                }
                _activation = std::move(activation);
            }
            // Get activation function
            const Activation& get_activation() const {
                return *_activation;
            }
        private:
            // Receive signal from synaptic connection
            void receive_signal(const SynapticConn* in_conn) {
                const auto found = std::any_of(std::begin(_in_connections), std::end(_in_connections), [&] (const auto &conn) { return conn == in_conn; });
                if (not found) {
                    throw std::invalid_argument("Neuron::receive_signal: invalid connection!");
                }
                _total_rx_signal += in_conn->get_weighted_signal();
                _number_rx_signals++;
                if (_number_rx_signals == _in_connections.size()) {
                    _weighted_sum = _total_rx_signal;
                    _signal = get_activation().value(_weighted_sum);
                    //std::println("Neuron::receive_signal: all signals received! Signal: {}, weighted sum: {}", _signal, _total_rx_signal);
                    broadcast_signal();
                    _number_rx_signals = 0;
                    _total_rx_signal = 0.0;
                }
            }
            // Update connection weight according to target and a learning rule
            virtual void update(SynapticConn* conn, const real_t target) = 0;
            // Data members
            size_t _number_rx_signals{0};
            real_t _total_rx_signal{0.0};
            real_t _weighted_sum{0.0};
            real_t _signal{0.0};
            std::vector<SynapticConn*> _in_connections{};
            std::vector<SynapticConn*> _out_connections{};
            std::unique_ptr<Activation> _activation{nullptr};
    };
    template <typename T>
        concept NeuronType = std::derived_from<T, Neuron>;
    template <typename T>
        concept NeuronPtr = std::indirectly_readable<std::decay_t<T>> and NeuronType<std::iter_value_t<std::decay_t<T>>>;
    template <typename T>
        concept NeuronRange = std::ranges::range<T> and NeuronPtr<std::ranges::range_value_t<T>>;
    template <typename T>
        concept NeuronView = std::ranges::view<T> and NeuronRange<T>;
}