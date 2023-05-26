import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from scipy.spatial.transform import Rotation as R
from myutils.utils import *
import open3d as o3d

if __name__ == "__main__":
    object_cls = objects['mug']
    object_cls.init_pose = (0, 0, 0)
    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    meshes = [coordinate, object_mesh]

    # Seg_pose estimated pose
    pose_path = '/home/shixu/My_env/Object_Pose_Estimation/segmentation-driven-pose/YCB-Video-Result/025_mug/'
    files = os.listdir(pose_path)
    files.sort()
    for i, file in enumerate(files[:5]):
        if i > 0:
            break
        pose_co = np.loadtxt(pose_path + file)
        r_co = pose_co[:, :3]
        t_co = pose_co[:, 3]
        r_oc = np.linalg.inv(r_co)
        t_oc = -np.linalg.inv(r_co).dot(t_co)
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05 * (i+1), origin=[0, 0, 0])
        camera.translate(t_oc, relative=True)
        camera.rotate(r_oc, center=t_oc)
        meshes.append(camera)

    # True pose
    r_co = [[0.8003, -0.5985, -0.0372],
            [-0.1667, -0.1625, -0.9725],
            [0.5760, 0.7845, -0.2298]]
    t_co = [-0.0296, -0.1095, 0.8431]
    r_oc = np.linalg.inv(r_co)
    t_oc = -np.linalg.inv(r_co).dot(t_co)
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    camera.translate(t_oc, relative=True)
    camera.rotate(r_oc, center=t_oc)
    meshes.append(camera)

    o3d.visualization.draw_geometries(meshes)
