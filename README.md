# gpu_utils

This repo contains utilities and tools for CUDA capable devices to avoid reinventing the wheel.

## Dependencies

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
* [gcc](https://gcc.gnu.org/)

## Installation

Copy-paste and execute the following lines in a terminal

```sh
git clone https://github.com/marvrez/gpu_utils.git
cd gpu_utils
make -j
```

## Querying the device

Execute the following to retrieve information about the device
```sh
./gpu_utils query
```
