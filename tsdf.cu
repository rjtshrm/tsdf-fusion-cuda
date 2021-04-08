#include "tsdf.h"



__global__ void fusion_gpu(Volume &vol, Views &views, float truncation_distance)
{
    int index = blockIdx.x * blockDim.x +  threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vox_res = 256 * 256 * 256;

    for(int vox_idx = index; vox_idx < vox_res; vox_idx += stride)
    {
        printf("1:: %d %d %d\n",vox_idx, vox_res, stride);
        int i, j, k;

        idx2ijk(vox_idx, vol.vol_dim, i, j, k);

        float x, y, z;

        ijk2xyz(i, j, k, vol.voxel_size, vol.origin_x, vol.origin_y, vol.origin_z, x, y, z);


        for (int idx = 0; idx < views.n_views; ++idx)
        {

            float _u, _v, _d;
            xyz2uv(idx, &views, x, y, z, _u, _v, _d);

            int u, v;
            u = int(u + 0.5);
            v = int(v + 0.5);

            if (u >= 0 && u < views.rows && v >= 0 && v < views.cols) {

                int depth_idx = (idx * views.rows + v) * views.cols + u;
                float depth = views.depth[depth_idx];
                float weight = views.weight[depth_idx];

                float depth_diff = depth - _d;

                float truncated_depth = fminf(1, fmaxf(-1, depth_diff / truncation_distance));

                if (depth_diff > 0 && truncated_depth >= -1) {
                    // add to volume
                    float new_weight =  vol.weight[vox_idx] + weight;

                    float new_value = (vol.data[vox_idx] * vol.weight[vox_idx] + truncated_depth * weight) / new_weight;
                    printf("ss %f", vol.data[0]);
                    //vol.data[vox_idx] = new_value;
                    //vol.weight[vox_idx] = new_weight;
                }

            }
        }
    }
}

void fusion(Volume &vol, Views &views, float truncation_distance)
{
    /**
     * Allocate memory for GPU
     */
    Volume vol_gpu;
    Views views_gpu;

    //mem_alloc_views(views_gpu, views_gpu);
    mem_alloc_volume(vol_gpu, vol);

    gpuErrchk(cudaGetLastError());
    printf("%f\n", vol.vol_dim);
    printf("%f\n", vol_gpu.vol_dim);
    printf("%f\n", vol.voxel_size);
    printf("%f\n", vol_gpu.voxel_size);
    printf("%f\n", vol.data[1]);
    printf("%f\n", vol.weight[1]);
    printf("%f\n", vol_gpu.data[0]);
    printf("%f\n", vol_gpu.weight[0]);

    fusion_gpu<<<getNumBlock(pow(vol_gpu.vol_dim, 3)), THREADS_PER_BLOCK>>>(vol_gpu, views_gpu, truncation_distance);
    //fusion_gpu<<<1, 1>>>(vol_gpu, views_gpu, truncation_distance);

    cudaDeviceSynchronize();

    //mem_free_views(views_gpu);
    //mem_free_volume(vol_gpu);

}
