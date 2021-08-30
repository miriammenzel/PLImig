#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>

namespace PLImg {
    class GPUOutOfMemoryException : std::exception {
        const char* what() const noexcept {return "Out of GPU memory. This can happen due to a Thrust call which tried to allocate too much memory.";}
    };

    class GPUExecutionException : std::exception {
        const char* what() const noexcept {return "CUDA error during CUDA call. This exception is not expected.";}
    };
}

#endif // EXCEPTIONS_H
