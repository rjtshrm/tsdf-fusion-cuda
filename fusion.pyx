import numpy as np
cimport numpy as np

np.import_array()



cdef extern from "tsdf.h":
    cdef cppclass Volume:
        float* data
        float *weight
        int vol_dim
        float voxel_size
        float origin_x
        float origin_y
        float origin_z

    cdef cppclass Views:
        int n_views
        float* depth
        float* weight
        int rows
        int cols
        float* K
        float* R
        float* T

    void fusion(Volume vol, Views views, float truncation_distance)




def fusion_tsdf(depth, weight, K, R, T, voxel_size=0.02, vol_dim=256, origin_x=0, origin_y=0, origin_z=0, truncation_factor=5):
    """
    :param depth: N*h*w depth maps
    :param weight: N*h*w corresponding weight for each depth maps
    :param K: N*3*3 intrisńsic matrix
    :param R: N*3*3 rotation matrix
    :param T: N*3*1 translation matrix
    :return:
    """

    cdef Views views
    cdef float[:, :, ::1] depth_view = depth.astype(np.float32)
    cdef float[:, :, ::1] weight_view = weight.astype(np.float32)
    cdef float[:, :, ::1] K_view = K.astype(np.float32)
    cdef float[:, :, ::1] R_view = R.astype(np.float32)
    cdef float[:, :, ::1] T_view = T.astype(np.float32)
    views.n_views = depth.shape[0]
    views.depth = &(depth_view[0, 0, 0])
    views.weight = &(weight_view[0, 0, 0])
    views.rows = depth.shape[1]
    views.cols = depth.shape[0]
    views.K = &(K_view[0, 0, 0])
    views.R = &(R_view[0, 0, 0])
    views.T = &(T_view[0, 0, 0])

    cdef Volume volume
    vol = np.ones((1, vol_dim, vol_dim, vol_dim), dtype=np.float32)
    _weight = np.ones((1, vol_dim, vol_dim, vol_dim), dtype=np.float32)
    cdef float[:, :, :, ::1] vol_view = vol
    cdef float[:, :, :, ::1] _weight_view = _weight
    volume.data = &(vol_view[0, 0, 0, 0])
    volume.weight = &(_weight_view[0, 0, 0, 0])
    volume.vol_dim = vol_dim
    volume.voxel_size = voxel_size
    volume.origin_x = origin_x
    volume.origin_y = origin_y
    volume.origin_z = origin_z

    fusion(volume, views, truncation_factor)


 
def print_hello(): 
    depth = np.empty((100, 20, 20))
    weight = np.empty((100, 20, 20))
    K = np.empty((100, 3, 3))
    R = np.empty((100, 3, 3)) 
    T = np.empty((100, 3, 1))    
    fusion_tsdf(depth, weight, K, R, T, voxel_size=0.02, vol_dim=256, origin_x=0, origin_y=0, origin_z=0,truncation_factor=5)