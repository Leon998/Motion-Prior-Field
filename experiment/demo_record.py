import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import load_mano
from myutils.object_config import objects
from myutils.assets_config import assets
from myutils.utils import *
from myutils.add_gauss_noise import add_gaussian_noise, noise_hand
import open3d as o3d
import torch
import redis
from can.hand_control import *
import keyboard, time
import math, random
from demo import *


if __name__ == "__main__":
    # ======================= experiment initialization ================================== #
    # Object name
    object_cls = objects['mustard_bottle']
    # pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)  
    # ====================== gpose prediction module initialization ======================== #
    poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
    model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')
    model.eval()

    # Coordinate
    device = "cuda"

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis')

    object = assets(mesh=object_cls.init_transform())
    vis.add_geometry(object.mesh)

    hand = assets(mesh=load_mano())
    vis.add_geometry(hand.mesh)

    pred_hand = assets(mesh=load_mano())
    pred_hand.mesh.paint_uniform_color([250 / 255, 127 / 255, 111 / 255])
    vis.add_geometry(pred_hand.mesh)

    while True:
        # hand
        t_wh = np.array([float(i) for i in r.get('hand_position')[1:-1].split(',')])
        q_wh = np.array([float(i) for i in r.get('hand_rotation')[1:-1].split(',')])
        # object
        t_wo = np.array([float(i) for i in r.get('object_position')[1:-1].split(',')])
        q_wo = np.array([float(i) for i in r.get('object_rotation')[1:-1].split(',')])
        object_pose_wdc = np.concatenate((q_wo, t_wo))

        q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
        hand_pose = np.concatenate((q_oh, t_oh), axis=0)
        
        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        pred_gpose = poses[idx]

        # ============================== update transform ============================= #
        noisy_hand_pose = noise_hand(hand_pose=hand_pose,std_q=0.01,std_t=0.01)
        noisy_pred_gpose = noise_hand(hand_pose=pred_gpose,std_q=0.01,std_t=0.005)
        hand.update_transform(noisy_hand_pose)
        pred_hand.update_transform(noisy_pred_gpose)

        vis.update_geometry(hand.mesh)
        vis.update_geometry(pred_hand.mesh)

        # 更新窗口
        time.sleep(0.05)
        p1= hand_pose[-3:]
        p2=math.sqrt(pow(p1[0],2)+pow(p1[1],2)+pow(p1[2],2))
        if p2 > 0.2:
            # 距离足够远才更新
            vis.poll_events()
            vis.update_renderer()

        # if keyboard.is_pressed('1'):
        #     wrist_tf(0, 40)
        # elif keyboard.is_pressed('2'):
        #     grasp_handle()
        #     release_grasp_long()
        # elif keyboard.is_pressed('3'):
        #     wrist_tf(-30, -45)
        # elif keyboard.is_pressed('4'):
        #     grasp_mug_top()
        #     release_grasp_long()
        #     init_pose()
