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
            void to_host(std::vector<NNet::State> &states) const {
                _backend.to_host(states);
            }
            void to_host(dataset_vector_t &outputs) {
                _backend.to_host(outputs);
            }
            // Update batch states using host weights.
            void update_states(const DataType auto &weights, index_t batch_begin = 0, index_t batch_end = -1) {
                _backend.update_states(weights, batch_begin, batch_end);
            }
        private:
            // Data members
            dag::update::omp::Backend _backend{};
    };
}
