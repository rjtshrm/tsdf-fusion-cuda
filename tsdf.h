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

inline int getNumBlock(int N) { return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; }

class Volume {
public:
    float *data;
    float *weight;
    int vol_dim;
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
inline void idx2ijk(int index, int res, int &i, int &j, int &k) {
    i = fmodf(index, res);
    j = fmodf((index - i) / res, res);
    k = index / (res * res);
}

GLOBAL_2_HOST
inline void ijk2xyz(int i, int j, int k, float voxel_size, float origin_x, float origin_y, float origin_z, float &x, float &y, float &z) {
    x = (i + 0.5) * voxel_size + origin_x;
    y = (j + 0.5) * voxel_size + origin_y;
    z = (k + 0.5) * voxel_size + origin_z;
}

GLOBAL_2_HOST
inline void xyz2uv(int idx, const Views *views, float x, float y, float z, float &u, float &v, float &d) {
    float *K = views->K + idx * 9;
    float *R = views->K + idx * 9;
    float *T = views->T + idx * 3;

    float xt = R[0] * x + R[1] * y + R[2] * z + T[0];
    float yt = R[3] * x + R[4] * y + R[5] * z + T[1];
    float zt = R[6] * x + R[7] * y + R[8] * z + T[2];

    u = K[0] * xt + K[1] * yt + K[2] * zt;
    v = K[3] * xt + K[4] * yt + K[5] * zt;
    d = K[6] * xt + K[7] * yt + K[8] * zt;

    u = u / d;
    v = v / d;

}


void mem_alloc_views(Views &views_gpu, Views &views_cpu) {
    views_gpu.n_views = views_cpu.n_views;
    views_gpu.rows = views_cpu.rows;
    views_gpu.cols = views_cpu.cols;
    int N = views_cpu.n_views * views_cpu.rows * views_cpu.cols;

    gpuErrchk(cudaMallocManaged(&views_gpu.depth, N * sizeof(float)));
    gpuErrchk(cudaMallocManaged(&views_gpu.weight, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(&views_gpu.depth, &views_cpu.depth, sizeof(views_cpu.depth), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&views_gpu.weight, &views_cpu.weight, sizeof(views_cpu.weight), cudaMemcpyHostToDevice));

    N = views_cpu.n_views * 3 * 3;
    gpuErrchk(cudaMallocManaged(&views_gpu.K, N * sizeof(float)));
    gpuErrchk(cudaMallocManaged(&views_gpu.R, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(&views_gpu.K, &views_cpu.K, N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&views_gpu.R, &views_cpu.R, N * sizeof(float), cudaMemcpyHostToDevice));

    N = views_cpu.n_views * 3;
    gpuErrchk(cudaMallocManaged(&views_gpu.T, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(&views_gpu.T, &views_cpu.T, N * sizeof(float), cudaMemcpyHostToDevice));
}

void host_2_device(const float *host, float *device, int N) {
    printf("%d", N);
    gpuErrchk(cudaMemcpy(device, host, N*sizeof(float), cudaMemcpyHostToDevice));
}

void device_malloc(float *device, int N) {
    gpuErrchk(cudaMalloc(&device, N*sizeof(float)));
}

void mem_alloc_volume(Volume &vol_gpu, Volume &vol_cpu) {
    int N = vol_cpu.vol_dim * vol_cpu.vol_dim * vol_cpu.vol_dim;
    device_malloc(vol_gpu.data, N);
    device_malloc(vol_gpu.weight, N);
    host_2_device(vol_cpu.data, vol_gpu.data, N);
    host_2_device(vol_cpu.weight, vol_gpu.weight, N);

    vol_gpu.vol_dim = vol_cpu.vol_dim;
    vol_gpu.voxel_size = vol_cpu.voxel_size;
    vol_gpu.origin_x = vol_gpu.origin_x;
    vol_gpu.origin_y = vol_gpu.origin_z;
    vol_gpu.origin_z = vol_gpu.origin_z;
}


void mem_free_views(Views &views_gpu) {
    cudaFree(&views_gpu.n_views);
    cudaFree(views_gpu.depth);
    cudaFree(views_gpu.weight);
    cudaFree(&views_gpu.rows);
    cudaFree(&views_gpu.cols);
    cudaFree(views_gpu.K);
    cudaFree(views_gpu.R);
    cudaFree(views_gpu.T);
}

void mem_free_volume(Volume &vol_gpu) {
    cudaFree(vol_gpu.data);
    cudaFree(&vol_gpu.weight);
    cudaFree(&vol_gpu.vol_dim);
    cudaFree(&vol_gpu.voxel_size);
    cudaFree(&vol_gpu.origin_x);
    cudaFree(&vol_gpu.origin_y);
    cudaFree(&vol_gpu.origin_z);
}

#endif //TSDF_GPU_TSDF_H
