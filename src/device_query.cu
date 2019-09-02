#include "device_query.h"
#include "cuda_runtime.h"

#include <stdio.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
        printf("%s", cudaGetErrorString(error)); \
    } \
  } while (0)

int device_query();
{
    int device;
    cudaDeviceProp prop;

    if (cudaGetDevice(&device) != cudaSuccess) {
        printf("No CUDA device present.");
        return 0;
    }

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device id:                     %d", device);
    printf("Major revision number:         %d", prop.major);
    printf("Minor revision number:         %d", prop.minor);
    printf("Name:                          %s", prop.name);
    printf("Total global memory:           %zu", prop.totalGlobalMem);
    printf("Total shared memory per block: %zu", prop.sharedMemPerBlock);
    printf("Total registers per block:     %d", prop.regsPerBlock);
    printf("Warp size:                     %d", prop.warpSize);
    printf("Maximum memory pitch:          %zu", prop.memPitch);
    printf("Maximum threads per block:     %d", prop.maxThreadsPerBlock;
    printf("Maximum dimension of block:    (%d, %d, %d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum dimension of grid:     (%d, %d, %d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); 
    printf("Clock rate:                    %d", prop.clockRate);
    printf("Total constant memory:         %zu", prop.totalConstMem);
    printf("Texture alignment:             %zu", prop.textureAlignment);
    printf("Concurrent copy and execution: %s", (prop.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d", prop.multiProcessorCount);
    printf("Kernel execution timeout:      %s", (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));

    return 1;
}
