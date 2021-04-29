#include "tsdf.h"

__global__ void fusion_gpu(Volume vol, Views views, float truncation_distance)
{
    int index = blockIdx.x * blockDim.x +  threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vox_res = vol.vol_dim_w * vol.vol_dim_h * vol.vol_dim_d;

    for(int vox_idx = index; vox_idx < vox_res; vox_idx += stride)
    {
        int i, j, k;
        idx2ijk(vox_idx, vol.vol_dim_w, vol.vol_dim_h, vol.vol_dim_d, i, j, k);

        float x, y, z;
        ijk2xyz(i, j, k, vol.voxel_size, vol.origin_x, vol.origin_y, vol.origin_z, x, y, z);

        //printf("idx=%d, i=%d, j=%d, k=%d, x=%f, y=%f, z=%f\n",vox_idx,  i, j, k, x, y, z);
        //printf("idx=%d, vidx=%d\n", vox_idx, ((vol.vol_dim + k) * vol.vol_dim + j) * vol.vol_dim + i);
        bool vol_idx_updated = false;
        for (int idx = 0; idx < views.n_views; ++idx)
        {

            float _u, _v, _z;
            xyz2uv(idx, &views, x, y, z, _u, _v, _z);

            int u, v;
            u = int(_u + 0.5);
            v = int(_v + 0.5);
            //if(k==103 && j==130 && i==153) printf("u=%d, v=%d, ur=%f, vr=%f\n", u,v, _u,_v);
            //printf("Debug:  %d %d %d %f %f %f %f %f %f %d %d\n", i, j, k, x, y, z, _u, _v, _d, u, v);

            //printf("%d %d\n", views.rows, views.cols);
            if (u >= 0 && u < views.cols && v >= 0 && v < views.rows) {

                int depth_idx = (idx * views.rows + v) * views.cols + u;
                float depth = views.depth[depth_idx];
                float weight = views.weight[depth_idx];

                float sdf = depth - _z;
                //if(k==103 && j==130 && i==153) printf("  dm_d=%f, dm_idx=%d, u=%d, v=%d, ur=%f, vr=%f\n", depth, depth_idx, u,v, _u, _v);


                //printf("ss %f %f\n", depth_diff, truncated_depth);
                if (depth > 0 && sdf >= -truncation_distance) {
                    float tsdf = fminf(1, fmaxf(-1, sdf / truncation_distance));
                    // add to volume
                    float new_weight =  vol.weight[vox_idx] + weight;
                    float new_value = (vol.data[vox_idx] * vol.weight[vox_idx] + tsdf * weight) / new_weight;

                    vol.data[vox_idx] = new_value;
                    vol.weight[vox_idx] = new_weight;
                    vol_idx_updated = true;
                }
            }
        }
        if (vol_idx_updated == false) {
            vol.data[vox_idx] = -1;
            vol.weight[vox_idx] = 0;
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

    mem_alloc_views_gpu(views_gpu, views);
    mem_alloc_volume_gpu(vol_gpu, vol);

    fusion_gpu<<<getNumBlock(vol_gpu.vol_dim_w * vol_gpu.vol_dim_h * vol_gpu.vol_dim_d, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(vol_gpu, views_gpu, truncation_distance);
    //fusion_gpu<<<1, 1>>>(vol_gpu, views_gpu, truncation_distance);

    cudaDeviceSynchronize();

    mem_alloc_volume_cpu(vol_gpu, vol);

    mem_free_views_gpu(views_gpu);
    mem_free_volume_gpu(vol_gpu);
}