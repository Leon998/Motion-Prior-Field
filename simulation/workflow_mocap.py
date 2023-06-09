import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import objects
from myutils.utils import *
import open3d as o3d
import torch
sys.path.append('pose_estimation/')
from pose_estimation.utils import *
from pose_estimation.utils4coordinate_tf import *
import redis


if __name__ == "__main__":  
    # pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)  
    # ====================== gpose prediction module initialization ======================== #
    object_cls = objects['mustard_bottle']
    poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
    model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')
    model.eval()

    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [coordinate, object_mesh]
    device = "cuda"

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis')
    vis.add_geometry(coordinate)
    vis.add_geometry(object_mesh)
    hand = load_mano()
    pred_hand = load_mano()
    pred_hand.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
    vis.add_geometry(hand)
    vis.add_geometry(pred_hand)
    translation = (0, 0, 0)
    rotation = (1, 0, 0, 0)
    pred_hand_translation = (0, 0, 0)
    pred_hand_rotation = (1, 0, 0, 0)
    while True:
        # hand
        t_wh = np.array([float(i) for i in r.get('hand_position')[1:-1].split(',')])
        q_wh = np.array([float(i) for i in r.get('hand_rotation')[1:-1].split(',')])
        # object
        t_wo = np.array([float(i) for i in r.get('object_position')[1:-1].split(',')])
        q_wo = np.array([float(i) for i in r.get('object_rotation')[1:-1].split(',')])

        q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
        hand_pose = np.concatenate((q_oh, t_oh), axis=0)

        translation, rotation, delta_translation, delta_R = incremental_hand_transform(hand, 
                                                                                   hand_pose, 
                                                                                   translation, 
                                                                                   rotation)

        hand.translate(delta_translation, relative=True)
        hand.rotate(delta_R, center=translation)

        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        pred_gpose = poses[idx]

        # pred_hand transform
        pred_hand_translation, pred_hand_rotation, pred_hand_delta_translation, pred_hand_delta_R = incremental_hand_transform(pred_hand, 
                                                                                   pred_gpose, 
                                                                                   pred_hand_translation, 
                                                                                   pred_hand_rotation)

        pred_hand.translate(pred_hand_delta_translation, relative=True)
        pred_hand.rotate(pred_hand_delta_R, center=pred_hand_translation)

        # ======================== wrist joint transformation ============================= #
        wrist_joint_zyx, r_transform = wrist_joint_transform(hand_pose, pred_gpose)
        print(wrist_joint_zyx)

        # transformed_hand = copy.deepcopy(hand)
        # transformed_hand.rotate(r_transform, center=hand_pose[4:])
        # transformed_hand.paint_uniform_color([255 / 255, 190 / 255, 122 / 255])

        # ========================= visualize in open3d ==================== #
        vis.update_geometry(hand)
        vis.update_geometry(pred_hand)

        # 更新窗口
        vis.poll_events()
        vis.update_renderer()

    # o3d.visualization.draw_geometries(meshes)

