import numpy as np
from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from myutils.utils import *
import open3d as o3d
import random


if __name__ == "__main__":
    object_cls = objects['mug']
    save_path = 'obj_coordinate/pcd_gposes/' + object_cls.name
    gposes_avg_path = save_path + '/' + 'gposes_label_avg_' + str(object_cls.g_clusters) + '.txt'
    field_path = 'obj_coordinate/pcd_field/' + object_cls.name

    # if gpose_avg is not appropriate for visualizing
    gposes_raw = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
    num_gtype = len(object_cls.grasp_types)
    file_indices = random.sample(range(0, len(gposes_raw)), num_gtype)  # 随便选的几个代表性gpose
    print(file_indices)
    gposes_raw = [gposes_raw[i] for i in file_indices]

    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [coordinate, object_mesh]

    gposes_label_avg = np.loadtxt(gposes_avg_path)
    gposes = gposes_label_avg[:, :]
    for i, gpose in enumerate(gposes_raw):
        hand_gtype = hand_transform(gpose, init_hand)
        # color_idx = i
        # hand_gtype.paint_uniform_color(colorlib[i//(len(file_indices)//(num_gtype))])
        meshes.append(hand_gtype)

    pcd = o3d.io.read_point_cloud(field_path + '/' + 'T_colored.xyzrgb')
    meshes.append(pcd)
    o3d.visualization.draw_geometries(meshes)