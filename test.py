import cv2 as cv
import numpy as np
import tsdf
from skimage import measure
import os


def export_off(vertices, triangles, filename):
    """
    Exports a mesh in the (.off) format.
    """
    with open(filename, 'w') as fh:
        fh.write('OFF\n')
        fh.write('{} {} 0\n'.format(len(vertices), len(triangles)))

        for v in vertices:
            fh.write("{} {} {}\n".format(*v))

        for f in triangles:
            fh.write("3 {} {} {}\n".format(*f))

def test_fusion():
    n = 100
    hgt = 640
    wdt = 640
    data_path = "data"
    depth = np.empty((n, hgt, wdt))
    weight = np.empty((n, hgt, wdt))
    K = np.empty((n, 3, 3))
    R = np.empty((n, 3, 3))
    T = np.empty((n, 3, 1))
    for i in range(n):
        depth[i, ...] = cv.imread(f"{data_path}/frame-{i:06}.depth.png", -1).astype('float') / 1000.
        weight = np.ones_like(depth)
        K[i, ...] = np.loadtxt(f"{data_path}/frame-{i:06}.intrinsic.txt")
        pose = np.loadtxt(f"{data_path}/frame-{i:06}.extrinsic.txt")
        R[i, ...] = pose[0:3, 0:3]
        T[i, ...] = pose[0:3, 3][:, np.newaxis]

    origin_x = -0.5
    origin_y = -0.5
    origin_z = -0.5
    resolution = 1 / 256
    vol, weight = tsdf.fusion_tsdf(depth, weight, K, R, T, voxel_size=resolution, vol_dim=(256, 256, 256), origin_x=origin_x, origin_y=origin_y, origin_z=origin_z, truncation_factor=2)


    verts, faces, normals, values = measure.marching_cubes(vol, 0)


    xyz = verts * resolution + [origin_x, origin_y, origin_z]
    export_off(xyz, faces, f"test.off")

    # Write header
    ply_file = open("test.ply", 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2]
        ))

if os.path.exists("test.off"):
    os.remove("test.off")
if os.path.exists("test.ply"):
    os.remove("test.ply")

test_fusion()