import os
import sys
sys.path.append(os.getcwd())
import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from simulation.tools import *
from scipy.spatial.transform import Rotation as R
from create_object import object_init
from myutils.object_config import objects
from myutils.utils import *
import torch

# 连接物理引擎
physicsCilent = p.connect(p.GUI)
device = "cuda"
# 渲染逻辑
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置环境重力加速度
p.setGravity(0, 0, 0)

rotate_frame = True
height = 0
if rotate_frame:
    # 加载URDF模型，此处是加载蓝白相间的陆地
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0.5*pi]))
    height = 0.67

# 加载机器人，并设置加载的机器人的位姿
robot_path = "simulation/wrist_hand_left_v2/urdf/wrist_hand_left_v2.urdf"
startPos = [0, 0, height]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_path, startPos, startOrientation, useFixedBase=1)
# 加载物体，并随机设置一个位姿
object_cls = objects['mug']
obj_path = object_cls.file_path
obj_startPos = [0.5, 0, height]  # 比较下来发现0.67的高度刚好在桌上
obj_startOrientation = p.getQuaternionFromEuler([0, 0, 0.5*pi])
obj = object_init(obj_path, q_init=obj_startOrientation, t_init=obj_startPos, p=p)
obj_state = p.getBasePositionAndOrientation(obj.object_id)
print("object state: ", obj_state)
# ====================== gpose prediction module initialization ======================== #
poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')
model.eval()


joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

p.setRealTimeSimulation(1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-90,
                                 cameraPitch=-37, cameraTargetPosition=obj_startPos)


Q_wh, T_wh, _, _, num_frame = read_data("simulation/trajectory/arm_000.csv")
i = 0
T_oh = np.zeros((1, 7))
while p.isConnected():
    time.sleep(1./240.)
    if i < num_frame:
        q_base, t_base = Q_wh[i], T_wh[i]
        q_wo, t_wo = obj_startOrientation, obj_startPos
        if rotate_frame:
            q_base, t_base = frame_rotation(q_base, t_base)
            t_base += np.array(startPos)
            q_wo = object_rotation(q_wo)
        p.resetBasePositionAndOrientation(robot_id, t_base, q_base)
        # 获取手部姿态
        hand_state = p.getLinkState(robot_id, 2)
        t_wh, q_wh = hand_state[0], hand_state[1]
        t_wh, q_wh, t_wo, q_wo = np.array(t_wh), np.array(q_wh), np.array(t_wo), np.array(q_wo)
        q_oh, t_oh, _ = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
        print("t_wh: ", q_wh, t_wh)
        print("t_wo: ", q_wo, t_wo)
        print("t_oh: ", q_oh, t_oh)
        hand_pose = np.concatenate((q_oh, t_oh), axis=0)
        T_oh =np.concatenate((T_oh, hand_pose.reshape(1,7)), axis=0)
        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        pred_gpose = poses[idx]
        euler_joint, r_transform = wrist_joint_transform(hand_pose, pred_gpose)
        print(euler_joint)

        i += 1
    np.savetxt("simulation/trajectory/T_oh", T_oh[1:])
    

# 断开连接
p.disconnect()