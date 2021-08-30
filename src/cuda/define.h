//
// Created by jreuter on 12.03.21.
//

#ifndef PLIMIG_DEFINE_H
#define PLIMIG_DEFINE_H

#define CHECK_CUDA(S) do { \
    cudaError_t e = S; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %d\n", __FILE__, __LINE__, e); \
        cudaDeviceReset(); \
        if(e == cudaErrorMemoryAllocation) { \
            throw PLImg::GPUOutOfMemoryException(); \
        } else { \
            throw PLImg::GPUExecutionException(); \
        } \
    } \
} while (false)

#define WHITE_VALUE 200
#define GRAY_VALUE 100

#endif //PLIMIG_DEFINE_H
