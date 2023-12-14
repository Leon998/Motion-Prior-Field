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

rotate_frame = False
# if rotate_frame:
#     # 加载URDF模型，此处是加载蓝白相间的陆地
#     planeId = p.loadURDF("plane.urdf")

# 加载机器人，并设置加载的机器人的位姿
robot_path = "simulation/wrist_hand_left_v2/urdf/wrist_hand_left_v2.urdf"
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_path, startPos, startOrientation, useFixedBase=1)

p.setRealTimeSimulation(1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=135,
                                 cameraPitch=-30, cameraTargetPosition=[0,0,0])

while p.isConnected():
    time.sleep(1./240.)

p.disconnect()