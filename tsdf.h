//
// Created by rajat on 10.03.21.
//

#ifndef TSDF_GPU_TSDF_H
#define TSDF_GPU_TSDF_H

#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#define GLOBAL_2_HOST __host__ __device__

#define gpuErrchk(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}


const int THREADS_PER_BLOCK = 1024;

inline int getNumBlock(int N, int threads_per_block) { return (N + threads_per_block - 1) / threads_per_block; }

class Volume {
public:
    float *data;
    float *weight;
    int vol_dim_w;
    int vol_dim_h;
    int vol_dim_d;
    float voxel_size;
    float origin_x;
    float origin_y;
    float origin_z;
};

class Views {
public:
    int n_views;
    float *depth;
    float *weight;
    int rows;
    int cols;
    float *K;
    float *R;
    float *T;
};

void fusion(Volume &vol, Views &views, float truncation_distance);

/**
 * transfer voxel coordinates to world coordinates
 */
GLOBAL_2_HOST
inline void idx2ijk(int index, int vol_dim_w, int vol_dim_h, int vol_dim_d, int &i, int &j, int &k) {
    i = index % vol_dim_w;
    j = ((index - i) / vol_dim_w) % vol_dim_h;
    k = index / (vol_dim_w * vol_dim_h);
}

GLOBAL_2_HOST
inline void ijk2xyz(int i, int j, int k, float voxel_size, float origin_x, float origin_y, float origin_z, float &x, float &y, float &z) {
    x = i * voxel_size + origin_x;
    y = j * voxel_size + origin_y;
    z = k * voxel_size + origin_z;
}

GLOBAL_2_HOST
inline void xyz2uv(int idx, const Views *views, float x, float y, float z, float &u, float &v, float &d) {
    float *K = views->K + idx * 9;
    float *R = views->R + idx * 9;
    float *T = views->T + idx * 3;

    float xt = R[0] * x + R[1] * y + R[2] * z + T[0];
    float yt = R[3] * x + R[4] * y + R[5] * z + T[1];
    float zt = R[6] * x + R[7] * y + R[8] * z + T[2];

    u = K[0] * xt + K[1] * yt + K[2] * zt;
    v = K[3] * xt + K[4] * yt + K[5] * zt;
    d = K[6] * xt + K[7] * yt + K[8] * zt;

    u = u / d;
    v = v / d;
    //printf("Projected coordinates %f %f %f\n", u, v, d);
}

void host_2_device(const float *host, float *device, int N) {
    gpuErrchk(cudaMemcpy(device, host, N*sizeof(float), cudaMemcpyHostToDevice));
}

void device_2_host(float *host, const float *device, int N) {
    gpuErrchk(cudaMemcpy(host, device, N*sizeof(float), cudaMemcpyDeviceToHost));
}

void mem_malloc_gpu(float **device, int N) {
    gpuErrchk(cudaMallocManaged(device, N*sizeof(float)));
}


void mem_alloc_views_gpu(Views &views_gpu, Views &views_cpu) {
    int N = views_cpu.n_views * views_cpu.rows * views_cpu.cols;
    mem_malloc_gpu(&views_gpu.depth, N);
    mem_malloc_gpu(&views_gpu.weight, N);
    host_2_device(views_cpu.depth, views_gpu.depth, N);
    host_2_device(views_cpu.weight, views_gpu.weight, N);

    N = views_cpu.n_views * 3 * 3;
    mem_malloc_gpu(&views_gpu.K, N);
    mem_malloc_gpu(&views_gpu.R, N);
    host_2_device(views_cpu.K, views_gpu.K, N);
    host_2_device(views_cpu.R, views_gpu.R, N);

    N = views_cpu.n_views * 3;
    mem_malloc_gpu(&views_gpu.T, N);
    host_2_device(views_cpu.T, views_gpu.T, N);

    views_gpu.n_views = views_cpu.n_views;
    views_gpu.rows = views_cpu.rows;
    views_gpu.cols = views_cpu.cols;
}

void mem_alloc_volume_gpu(Volume &vol_gpu, Volume &vol_cpu) {
    int N = vol_cpu.vol_dim_w * vol_cpu.vol_dim_h * vol_cpu.vol_dim_d;
    mem_malloc_gpu(&vol_gpu.data, N);
    mem_malloc_gpu(&vol_gpu.weight, N);

    host_2_device(vol_cpu.data, vol_gpu.data, N);
    host_2_device(vol_cpu.weight, vol_gpu.weight, N);

    vol_gpu.vol_dim_w = vol_cpu.vol_dim_w;
    vol_gpu.vol_dim_h = vol_cpu.vol_dim_h;
    vol_gpu.vol_dim_d = vol_cpu.vol_dim_d;
    vol_gpu.voxel_size = vol_cpu.voxel_size;
    vol_gpu.origin_x = vol_cpu.origin_x;
    vol_gpu.origin_y = vol_cpu.origin_z;
    vol_gpu.origin_z = vol_cpu.origin_z;
}

void mem_alloc_volume_cpu(Volume &vol_gpu, Volume &vol_cpu) {
    int N = vol_gpu.vol_dim_w * vol_gpu.vol_dim_h * vol_gpu.vol_dim_d;

    device_2_host(vol_cpu.data, vol_gpu.data, N);
    device_2_host(vol_cpu.weight, vol_gpu.weight, N);

    vol_cpu.vol_dim_w = vol_gpu.vol_dim_w;
    vol_cpu.vol_dim_h = vol_gpu.vol_dim_h;
    vol_cpu.vol_dim_d = vol_gpu.vol_dim_d;
    vol_cpu.voxel_size = vol_gpu.voxel_size;
    vol_cpu.origin_x = vol_gpu.origin_x;
    vol_cpu.origin_y = vol_gpu.origin_z;
    vol_cpu.origin_z = vol_gpu.origin_z;
}

void mem_free_views_gpu(Views &views_gpu) {
    cudaFree(views_gpu.depth);
    cudaFree(views_gpu.weight);
    cudaFree(views_gpu.K);
    cudaFree(views_gpu.R);
    cudaFree(views_gpu.T);
}

void mem_free_volume_gpu(Volume &vol_gpu) {
    cudaFree(vol_gpu.data);
    cudaFree(vol_gpu.weight);
}

#endif //TSDF_GPU_TSDF_H
