#include "device_query.h"
#include "logger.h"
#include "cuda_runtime.h"

#include <assert.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
        LOG_ERROR("%s", cudaGetErrorString(error)); \
    } \
  } while (0)

int device_query();
{
    int device;
    cudaDeviceProp prop;

    if (cudaGetDevice(&device) != cudaSuccess) {
        LOG_ERROR("No CUDA device present.");
        return 0;
    }

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    LOG_INFO("Device id:                     %d", device);
    LOG_INFO("Major revision number:         %d", prop.major);
    LOG_INFO("Minor revision number:         %d", prop.minor);
    LOG_INFO("Name:                          %s", prop.name);
    LOG_INFO("Total global memory:           %zu", prop.totalGlobalMem);
    LOG_INFO("Total shared memory per block: %zu", prop.sharedMemPerBlock);
    LOG_INFO("Total registers per block:     %d", prop.regsPerBlock);
    LOG_INFO("Warp size:                     %d", prop.warpSize);
    LOG_INFO("Maximum memory pitch:          %zu", prop.memPitch);
    LOG_INFO("Maximum threads per block:     %d", prop.maxThreadsPerBlock;
    LOG_INFO("Maximum dimension of block:    (%d, %d, %d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    LOG_INFO("Maximum dimension of grid:     (%d, %d, %d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); 
    LOG_INFO("Clock rate:                    %d", prop.clockRate);
    LOG_INFO("Total constant memory:         %zu", prop.totalConstMem);
    LOG_INFO("Texture alignment:             %zu", prop.textureAlignment);
    LOG_INFO("Concurrent copy and execution: %s", (prop.deviceOverlap ? "Yes" : "No"));
    LOG_INFO("Number of multiprocessors:     %d", prop.multiProcessorCount);
    LOG_INFO("Kernel execution timeout:      %s", (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));

    return 1;
}
