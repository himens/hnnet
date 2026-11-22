#include <metal_stdlib>

// Names used by the activation headers shared with the CPU
namespace hnnet {
    using real_t = float;
}
namespace std {
    using ::metal::exp;
    using ::metal::log;
    using ::metal::max;
    using ::metal::pow;
    using ::metal::tanh;
    inline float abs(const float x) {
        return ::metal::fabs(x);
    }
}
