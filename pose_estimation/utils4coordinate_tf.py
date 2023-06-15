import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def cam2handpose(pose_co, init_pose):
    mocap_frame = list(init_pose)
    cam_frame = [item * (180/np.pi) for item in mocap_frame]
    # mocap中init_pose的顺序是xyz，但是采集时y和z坐标相当于对调了一下，因此这里是xzy
    r_mocap2cam = R.from_euler('xzy', cam_frame, degrees=True).as_matrix()
    r_co = pose_co[:, :3]
    t_co = pose_co[:, 3]
    # print(r_co, t_co)
    r_oc = np.linalg.inv(r_co)
    t_oc = -np.linalg.inv(r_co).dot(t_co)
    # rotation from pose estimation to Mocap
    r_oc = r_mocap2cam.dot(r_oc)
    t_oc = r_mocap2cam.dot(t_oc)

    # 变换camera pose到hand pose
    T_ch = np.loadtxt("pose_estimation/hand_eye_calibrate/T_ch.txt")
    r_ch = T_ch[:, :-1]
    t_ch = T_ch[:, -1]
    r_oh = r_oc.dot(r_ch)
    t_oh = r_oc.dot(t_ch) + t_oc
    q_oh = R.from_matrix(r_oh).as_quat()
    hand_pose = np.concatenate((q_oh.reshape(1, -1), t_oh.reshape(1, -1)), axis=1).reshape(-1)
    return hand_pose, r_oc, t_oc

def wrist_joint_transform(current_hand_pose, gpose):
    r_mid_pose = R.from_quat(current_hand_pose[:4]).as_matrix()
    r_target_pose = R.from_quat(gpose[:4]).as_matrix()
    r_transform = (r_target_pose).dot(np.linalg.inv(r_mid_pose))  # 用来可视化变换
    r_joint = np.linalg.inv(r_mid_pose).dot(r_target_pose)  # 用来计算关节变换角
    euler_joint = R.from_matrix(r_joint).as_euler('zyx', degrees=True)
    return euler_joint, r_transform