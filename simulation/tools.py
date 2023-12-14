import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os


def read_data_sim(file_name):
    """
    Extract rotation (quaternion type) and translation, and transform into numpy array
    Parameters
    ----------
    file_name : name of csv file
    Returns
    ----------
    Q_wh, T_wh, Q_wo, T_wo, num_frame : quaternion and translation, stack vertically by time
        In shape of (num_frame, 4), (num_frame, 3), (num_frame, 4), (num_frame, 3)
    Mocap传过来的数据，坐标系是Y轴朝上。
    """
    df_raw = pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8], skiprows=6)
    data_raw = np.array(df_raw)
    num_frame = data_raw.shape[0]
    Q_wh = data_raw[:, :4]  # X,Y,Z,W
    T_wh = data_raw[:, 4:7]  # X,Y,Z
    return Q_wh, T_wh, num_frame



# def rotation_transform(q_wh, startOrientation):
#     """
#     将机械手的初始旋转矩阵施加到每个时刻中
#     """

#     init_R = R.from_quat(startOrientation).as_matrix()
#     current_R = R.from_quat(q_wh).as_matrix()
#     real_R = current_R.dot(init_R)
#     real_q = R.from_matrix(real_R).as_quat()
#     return real_q

def frame_rotation(q_base, t_base):
    """
    绕x轴旋转整个坐标系90°
    """
    r_base = R.from_quat(q_base).as_matrix()
    r_frame = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    new_r_base = r_frame.dot(r_base)
    new_t_base = r_frame.dot(t_base)
    new_q_base = R.from_matrix(new_r_base).as_quat()
    return new_q_base, new_t_base

def object_rotation(q_wo):
    """
    物体绕自身x轴旋转90°
    """
    r_wo = R.from_quat(q_wo).as_matrix()
    r_self = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    new_r_wo = r_wo.dot(r_self)
    new_q_wo = R.from_matrix(new_r_wo).as_quat()
    return new_q_wo

def auto_control(robot_id, p, euler_joint):
    maxV = [2, 2, 2]
    maxF = [0.1, 0.2, 0.2]
    p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[0, 1, 2],
            controlMode=p.POSITION_CONTROL,
            targetPositions=euler_joint,
            targetVelocities=maxV,
            forces=maxF
            )

def kbd_controller(robot_id, p):
    maxV = 2
    maxF = 0.01
    key_dict = p.getKeyboardEvents()
    if len(key_dict):
        if p.B3G_LEFT_ARROW in key_dict:  # 左旋腕
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=[0],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[maxV],
                forces=[maxF]
            )
        elif p.B3G_RIGHT_ARROW in key_dict:  # 右旋腕
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=[0],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-maxV],
                forces=[maxF]
            )
        elif p.B3G_DELETE in key_dict:  # 左切腕
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=[1],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[maxV],
                forces=[maxF]
            )
        elif p.B3G_END in key_dict:  # 右切腕
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=[1],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-maxV],
                forces=[maxF]
            )
        elif p.B3G_UP_ARROW in key_dict:  # 上翻腕
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=[2],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-maxV],
                forces=[maxF]
            )
        elif p.B3G_DOWN_ARROW in key_dict:  # 下翻腕
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=[2],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[maxV],
                forces=[maxF]
            )
    else:           # 没有按键，则停下
        p.setJointMotorControlArray(   
            bodyUniqueId=robot_id,
            jointIndices=[0, 1, 2],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0, 0, 0],
            forces=[10, 10, 10]
        )
