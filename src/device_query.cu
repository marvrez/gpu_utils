#include "device_query.h"
#include "cuda_runtime.h"

#include <stdio.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
        printf("%s\n", cudaGetErrorString(error)); \
    } \
  } while (0)

int device_query()
{
    int device;
    cudaDeviceProp prop;

    if (cudaGetDevice(&device) != cudaSuccess) {
        printf("No CUDA device present.\n");
        return 0;
    }

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device id:                     %d\n", device);
    printf("Major revision number:         %d\n", prop.major);
    printf("Minor revision number:         %d\n", prop.minor);
    printf("Name:                          %s\n", prop.name);
    printf("Total global memory:           %zu\n", prop.totalGlobalMem);
    printf("Total shared memory per block: %zu\n", prop.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", prop.regsPerBlock);
    printf("Warp size:                     %d\n", prop.warpSize);
    printf("Maximum memory pitch:          %zu\n", prop.memPitch);
    printf("Maximum threads per block:     %d\n", prop.maxThreadsPerBlock);
    printf("Maximum dimension of block:    (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum dimension of grid:     (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); 
    printf("Clock rate:                    %d\n", prop.clockRate);
    printf("Total constant memory:         %zu\n", prop.totalConstMem);
    printf("Texture alignment:             %zu\n", prop.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (prop.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", prop.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));

    return 1;
}
