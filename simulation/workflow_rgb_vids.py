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
from pose_estimation.segpose_net import SegPoseNet
from pose_estimation.pred_img import pred_pose
import cv2


if __name__ == "__main__":
    # ====================== Pose eatimation module initialization ======================== #
    use_gpu = True
    # intrinsics
    k_ycbvideo = np.array([[0.338569200e+03, 0.00000000e+00, 3.17869800e+02],
                           [0.00000000e+00, 0.338800400e+03, 2.45317800e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # 21 objects for YCB-Video dataset
    object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
    vertex_ycbvideo = np.load('pose_estimation/data/YCB-Video/YCB_vertex.npy')

    # Loading segpose model
    data_cfg = 'pose_estimation/data/data-YCB.cfg'
    weightfile = 'pose_estimation/model/ycb-video.pth'
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    # ====================== gpose prediction module initialization ======================== #
    object_cls = objects['potted_meat_can']
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

    # capture = cv2.VideoCapture('pose_estimation/videos/mug523.mp4')
    capture = cv2.VideoCapture(0)
    i = 0

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis', width=1080, height=720)
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


    while capture.isOpened():
        # =========================== pose estimation=============================== #
        ret, imgfile = capture.read()
        if not ret:
            break
        pose_co, detect_flag = pred_pose(m, imgfile, object_names_ycbvideo, object_cls.name, k_ycbvideo,
                             vertex_ycbvideo, bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu, vis=False)
        print(pose_co)

        if detect_flag:
            # =========================== coordinate transformastion =============================== #
            hand_pose, r_oc, t_oc = cam2handpose(pose_co, object_cls.init_pose)

            # hand transform
            translation, rotation, delta_translation, delta_R = incremental_hand_transform(hand, 
                                                                                   hand_pose, 
                                                                                   translation, 
                                                                                   rotation)

            hand.translate(delta_translation, relative=True)
            hand.rotate(delta_R, center=translation)

            # camera
            # camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # camera.translate(t_oc, relative=True)
            # camera.rotate(r_oc, center=t_oc)

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
            # time.sleep(0.1)

        i += 1

    # o3d.visualization.draw_geometries(meshes)

