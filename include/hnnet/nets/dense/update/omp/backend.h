#pragma once
#include <span>
#include "hnnet/nets/dense/dense_forward_net.h"
#include "hnnet/nets/dag/update/omp/backend.h"

namespace hnnet::dense::update::omp {
    class Backend {
        public:
            // Initialize the dense update backend.
            void init(net::DenseForwardNet::View view) {
                _backend.init(view);
            }
            // Transfer data to the backend.
            void to_device(const DatasetType auto &inputs) {
                _backend.to_device(inputs);
            }
            // Transfer data to the host.
            void to_host(std::span<const NNet::State> &states) const {
                _backend.to_host(states);
            }
            void to_host(dataset_vector_t &outputs) {
                _backend.to_host(outputs);
            }
            // Update states using host weights.
            void update_states(const DataType auto &weights, index_t input_index = -1) {
                _backend.update_states(weights, input_index);
            }
        private:
            // Data members
            dag::update::omp::Backend _backend{};
    };
}
