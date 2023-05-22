import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from myutils.utils import *
import open3d as o3d
import torch
sys.path.append('pose_estimation/')
from pose_estimation.utils import *
from pose_estimation.utils4coordinate_tf import *
from pose_estimation.segpose_net import SegPoseNet
from pose_estimation.pred_img import pred_pose


if __name__ == "__main__":
    # ====================== Pose eatimation module initialization ======================== #
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
    vertex_ycbvideo = np.load('pose_estimation/data/YCB-Video/YCB_vertex.npy')

    # Loading segpose model
    data_cfg = 'pose_estimation/data/data-YCB.cfg'
    weightfile = 'pose_estimation/model/ycb-video.pth'
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    # ====================== gpose prediction module initialization ======================== #
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

    # Reading images
    file_path = 'pose_estimation/real_ycb.txt'
    with open(file_path, 'r') as file:
        imglines = file.readlines()
    # print(imglines)

    for i in range(len(imglines)):  # 只取前3张图
        # =========================== pose estimation=============================== #
        imgfile = imglines[i].rstrip()
        print(imgfile)
        raw_img = cv2.imread(imgfile)
        pose_co, detect_flag = pred_pose(m, raw_img, object_names_ycbvideo, object_cls.name, k_ycbvideo,
                             vertex_ycbvideo, bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)
        print("detect_flag:", detect_flag)
        print("object pose:", pose_co)

        if detect_flag:
            # =========================== coordinate transformastion =============================== #
            current_hand_pose, r_oc, t_oc = cam2handpose(pose_co)

            start_hand = hand_transform(current_hand_pose, init_hand)
            meshes.append(start_hand)
        
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            camera.translate(t_oc, relative=True)
            camera.rotate(r_oc, center=t_oc)
            meshes.append(camera)

            # ======================== grasp pose prediction ============================= #
            x = torch.from_numpy(current_hand_pose).type(torch.FloatTensor).to(device)
            pred = model(x)
            idx = pred.argmax(0).item()
            gpose = poses[idx]
            pred_hand = hand_transform(gpose, init_hand)
            pred_hand.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
            pred_hand.scale(0.8, center=pred_hand.get_center())
            meshes.append(pred_hand)

            # ======================== wrist joint transformation ============================= #
            wrist_joint_zyx, r_transform = wrist_joint_transform(current_hand_pose, gpose)
            print(wrist_joint_zyx)

            transformed_hand = copy.deepcopy(start_hand)
            transformed_hand.rotate(r_transform, center=current_hand_pose[4:])
            transformed_hand.paint_uniform_color([255 / 255, 190 / 255, 122 / 255])
            meshes.append(transformed_hand)

    o3d.visualization.draw_geometries(meshes)

