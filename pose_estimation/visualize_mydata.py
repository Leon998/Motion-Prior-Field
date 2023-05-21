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
    # Rotation between Mocap and pose estimation
    # 这是因为Mocap时，将物体坐标系绕x轴旋转了90度，所以在位姿估计的时候要变换一次
    r_mocap2cam = R.from_euler('x', -90, degrees=True).as_matrix()

    pose_path = '/home/shixu/My_env/Object_Pose_Estimation/segmentation-driven-pose/Real-YCB-Img-Out/025_mug/'
    all_files = os.listdir(pose_path)
    all_files.sort()
    files = all_files[:4]
    print(files)
    for i, file in enumerate(files):
        pose_co = np.loadtxt(pose_path + file)
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

        # Prediction
        # q_oc = R.from_matrix(r_oc).as_quat()
        # camera_pose = np.concatenate((q_oc.reshape(1, -1), t_oc.reshape(1, -1)), axis=1).reshape(-1)
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
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        gpose = poses[idx]
        pred_gpose = hand_transform(gpose, init_hand)
        pred_gpose.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
        pred_gpose.scale(0.5 + 0.2*i, center=pred_gpose.get_center())
        meshes.append(pred_gpose)

    o3d.visualization.draw_geometries(meshes)

