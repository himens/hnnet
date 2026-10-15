#pragma once
#include "hnnet/builtin/backends/omp/backprop/backend.h"
#include "hnnet/builtin/backends/omp/backprop/optimizers/momentum.h"
#include "hnnet/builtin/backends/omp/backprop/optimizers/rprop.h"

namespace hnnet::builtin::backend::omp::backprop {
    using Momentum = Backend<optimizer::Momentum>;
    using Rprop = Backend<optimizer::Rprop>;
}
