#pragma once
#include "hnnet/activation.h"

// TODO: 1) To increase performance in receive_signal(), check on incoming connection can be removed --> receive_signal(const real_t signal)?

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
                // Transmit signal
                template <typename RxType = Neuron, typename ReceivePtr>
                    //requires std::is_member_function_pointer_v<ReceivePtr> and std::is_invocable_v<ReceivePtr, RxType, real_t>
                    void send(const real_t data, ReceivePtr receive) const {
                        if ((rx == nullptr) or (tx == nullptr)) {
                            throw std::invalid_argument("SynapticConn::transmit_signal: nullptr rx or tx!");
                        }
                        const auto weighted_data = data * weight;
                        (static_cast<RxType*>(rx)->*receive)(weighted_data);
                    }
                // Data members
                //const Neuron* const tx{nullptr};
                Neuron* const tx{nullptr};
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
            // Get net signal (sum of all inputs from network)
            real_t get_net_signal() const {
                return _net_signal;
            }
            // Broadcast signal to all receiver neurons
            void broadcast_signal() const {
                for (const auto &conn : _out_connections) {
                    conn->send(_signal, &Neuron::receive_signal);
                }
            }
            // Add connection to neuron
            void add_connection(SynapticConn* conn) {
                if (conn == nullptr) {
                    throw std::invalid_argument("Neuron::add_connection: nullptr connection!");
                }
                if ((conn->rx == nullptr) or (conn->tx == nullptr)) {
                    throw std::invalid_argument("Neuron::add_connection: nullptr rx or tx!");
                }
                if ((conn->tx != this) and (conn->rx != this)) {
                    throw std::invalid_argument("Neuron::add_connection: invalid connection!");
                }
                const auto found = std::any_of(std::begin(_in_connections),  std::end(_in_connections),  [&] (const auto &in_conn) { return in_conn == conn; }) or
                                   std::any_of(std::begin(_out_connections), std::end(_out_connections), [&] (const auto &out_conn) { return out_conn == conn; });
                if (found) {
                    throw std::invalid_argument("Neuron::add_connection: duplicate connection!");
                }
                if (conn->tx == this) {
                    _out_connections.push_back(conn);
                }
                if (conn->rx == this) {
                    _in_connections.push_back(conn);
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
            // Learn from error: spread knowledge and update connection weights
            virtual void learn(const real_t error) = 0;
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
            void receive_signal(const real_t signal) {
                //const auto found = std::any_of(std::begin(_in_connections), std::end(_in_connections), [&] (const auto &conn) { return conn == in_conn; });
                //if (not found) {
                //    throw std::invalid_argument("Neuron::receive_signal: invalid connection!");
                //}
                _total_rx_signal += signal;
                _number_rx_signals++;
                if (_number_rx_signals == _in_connections.size()) {
                    _net_signal = _total_rx_signal;
                    _signal = _activation->value(_net_signal);
                    //std::println("Neuron::receive_signal: all signals received! Signal: {}, weighted sum: {}", _signal, _total_rx_signal);
                    broadcast_signal();
                    _number_rx_signals = 0;
                    _total_rx_signal = 0.0;
                }
            }
            // Data members
            size_t _number_rx_signals{0};
            real_t _total_rx_signal{0.0};
            real_t _net_signal{0.0};
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