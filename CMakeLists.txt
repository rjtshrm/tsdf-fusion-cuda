cmake_minimum_required(VERSION 3.2)

project(tsdf_gpu LANGUAGES CXX CUDA)

find_package(CUDA)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -Wall")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -gencode=arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -gencode=arch=compute_50,code=compute_50")

set(CMAKE_CUDA_STANDARD 11)

set(FUSION_GPU_SRC
        tsdf.cu
        )

cuda_add_library(tsdf_gpu SHARED ${FUSION_GPU_SRC})

set_target_properties(
        tsdf_gpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)