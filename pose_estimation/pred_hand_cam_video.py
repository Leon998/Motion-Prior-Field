import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import objects
from myutils.utils import *
import torch
from pose_estimation.utils import *
from pose_estimation.segpose_net import SegPoseNet
from pose_estimation.pred_img import pred_pose


if __name__ == "__main__":
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
    object_cls = objects['mug']
    poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
    model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')
    model.eval()

    device = "cuda"

    capture = cv2.VideoCapture("pose_estimation/videos/717.mp4")

    while True:
        # =========================== pose estimation=============================== #
        ret, imgfile = capture.read()
        if not ret:
            break
        pose_co, detect_flag = pred_pose(m, imgfile, object_names_ycbvideo, object_cls.name, k_ycbvideo,
                             vertex_ycbvideo, bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu, vis=True)
        print(pose_co)