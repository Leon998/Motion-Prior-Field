from utils import *
from segpose_net import SegPoseNet
import cv2
from pred_img import pred_pose

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from scipy.spatial.transform import Rotation as R
from myutils.utils import *
import open3d as o3d
import torch


use_gpu = True
# intrinsics
k_ycbvideo = np.array([[0.338856700e+03, 0.00000000e+00, 3.12340100e+02],
                               [0.00000000e+00, 0.339111500e+03, 2.46983900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# 21 objects for YCB-Video dataset
object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
target_object = '025_mug'
vertex_ycbvideo = np.load('pose_estimation/data/YCB-Video/YCB_vertex.npy')
# ======================================= #
# Rotation between Mocap and pose estimation
# 这是因为Mocap时，将物体坐标系绕x轴旋转了90度，所以在位姿估计的时候要变换一次
r_mocap2cam = R.from_euler('x', -90, degrees=True).as_matrix()

object_cls = objects['mug']
poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/noisy_diverse.pkl')
model.eval()

# Coordinate
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
# Object
object_mesh = object_cls.init_transform()
# Hand
init_hand = load_mano()
meshes = [coordinate, object_mesh]
device = "cuda"

if __name__ == "__main__":
    file_path = 'pose_estimation/real_ycb.txt'
    with open(file_path, 'r') as file:
        imglines = file.readlines()
    # print(imglines)
    
    # Loading segpose model
    data_cfg = 'pose_estimation/data/data-YCB.cfg'
    weightfile = 'pose_estimation/model/ycb-video.pth'
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    for i in range(len(imglines[:3])):
        # =========================== pose estimation=============================== #
        imgfile = imglines[i].rstrip()
        print(imgfile)
        pose_co = pred_pose(m, imgfile, object_names_ycbvideo, target_object, k_ycbvideo,
                             vertex_ycbvideo, bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)
        print(pose_co)

        # =========================== coordinate transformastion =============================== #
        r_co = pose_co[:, :3]
        t_co = pose_co[:, 3]
        # print(r_co, t_co)
        r_oc = np.linalg.inv(r_co)
        t_oc = -np.linalg.inv(r_co).dot(t_co)
        # rotation from pose estimation to Mocap
        r_oc = r_mocap2cam.dot(r_oc)
        t_oc = r_mocap2cam.dot(t_oc)
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 + 0.02*i, origin=[0, 0, 0])
        camera.translate(t_oc, relative=True)
        camera.rotate(r_oc, center=t_oc)
        meshes.append(camera)

        # 变换camera pose到hand pose
        T_ch = np.loadtxt("pose_estimation/hand_eye_calibrate/T_ch.txt")
        r_ch = T_ch[:, :-1]
        t_ch = T_ch[:, -1]
        r_oh = r_oc.dot(r_ch)
        t_oh = r_oc.dot(t_ch) + t_oc
        q_oh = R.from_matrix(r_oh).as_quat()
        hand_pose = np.concatenate((q_oh.reshape(1, -1), t_oh.reshape(1, -1)), axis=1).reshape(-1)

        start_hand = hand_transform(hand_pose, init_hand)
        meshes.append(start_hand)

        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        gpose = poses[idx]
        pred_hand = hand_transform(gpose, init_hand)
        pred_hand.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
        pred_hand.scale(0.5 + 0.2*i, center=pred_hand.get_center())
        meshes.append(pred_hand)

        # ======================== wrist joint transformation ============================= #
        r_mid_pose = R.from_quat(hand_pose[:4]).as_matrix()
        r_target_pose = R.from_quat(gpose[:4]).as_matrix()
        r_transform = (r_target_pose).dot(np.linalg.inv(r_mid_pose))
        # q_transform = R.from_matrix(r_transform).as_quat()
        euler_transform = R.from_matrix(r_transform).as_euler('zyx', degrees=True)
        print(euler_transform)

        transformed_hand = copy.deepcopy(start_hand)
        transformed_hand.rotate(r_transform, center=hand_pose[4:])
        transformed_hand.paint_uniform_color([255 / 255, 190 / 255, 122 / 255])
        meshes.append(transformed_hand)

    o3d.visualization.draw_geometries(meshes)
