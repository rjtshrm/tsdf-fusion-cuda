cmake_minimum_required(VERSION 3.2)

project(tsdf_gpu LANGUAGES CXX CUDA)

find_package(CUDA)

find_package (Eigen3 3 REQUIRED NO_MODULE)

set(CMAKE_CUDA_STANDARD 14)

set(FUSION_GPU_SRC
        tsdf.cu
        )

cuda_add_library(tsdf_gpu SHARED ${FUSION_GPU_SRC})

set_target_properties(
        tsdf_gpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)