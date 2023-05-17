from utils.hand_config import *
from utils.object_config import colorlib, objects
from utils.utils import *
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

gpose_debug = 1


def save_gpose_avg(object_cls, dubug=False):
    gtypes = object_cls.grasp_types
    save_path = 'obj_coordinate/pcd_gposes/' + object_cls.name
    gposes_path = save_path + '/' + 'gposes_raw.txt'
    gpose_label_path = save_path + '/' + 'gposes_label.txt'
    gposes_raw = np.loadtxt(gposes_path)
    gpose_label = np.loadtxt(gpose_label_path)
    # label = []  # 4-number system transform
    # for l in gpose_label:
    #     tmp = int(l[0] * object_cls.g_clusters + l[1])
    #     label.append(tmp)
    item = np.unique(gpose_label)  # unique gtype number, [0, 1, 2, ..., 11]
    index_list = np.arange(len(gpose_label))  # index list, the index of files [0, 1, 2, ...,299]
    gpose_avg = []
    for i, g in enumerate(item):
        g_index = index_list[gpose_label == g]
        gposes = gposes_raw[g_index]
        g_avg = rotation_avg(gposes)
        if dubug:
            if i == gpose_debug:
                print(g_index)
                print(gposes.shape)
                print(g_avg)
                gpose_visualize(gposes, g_avg)
        gpose_avg.append(g_avg)
    gposes_avg = np.array(gpose_avg)
    # print(gposes_label_avg.shape)
    cls_idx = np.array(item).reshape(len(item), 1)
    # print(cls_idx)
    # gposes_avg = np.concatenate((gposes_avg, cls_idx), axis=1)
    np.savetxt(save_path + '/' + 'gposes_label_avg_' + str(object_cls.g_clusters) + '.txt', gposes_avg)


def rotation_avg(gposes):
    g_avg = np.mean(gposes, axis=0)
    return g_avg


def gpose_visualize(g_gposes, g_avg):
    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [coordinate, object_mesh]

    for i, gpose in enumerate(g_gposes):
        hand_gtype = hand_transform(gpose, init_hand)
        meshes.append(hand_gtype)

    avg_hand = hand_transform(g_avg, init_hand)
    avg_hand.paint_uniform_color(colorlib[gpose_debug])
    meshes.append(avg_hand)
    o3d.visualization.draw_geometries(meshes)


if __name__ == "__main__":
    object_cls = objects['mug']
    save_gpose_avg(object_cls, dubug=True)
