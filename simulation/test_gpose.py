import os
import sys
sys.path.append(os.getcwd())
import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from simulation.tools import *
from simulation.controller import auto_controller, kbd_controller
from scipy.spatial.transform import Rotation as R
from create_object import object_init
from myutils.object_config import objects
from myutils.utils import *
import torch
import keyboard

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
robot_path = "simulation/hand_left_v2/urdf/hand_left_v2.urdf"
startPos = [0, 0, height]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_path, startPos, startOrientation, useFixedBase=1)
# 加载物体，并随机一个位姿
object_cls = objects['mug']
obj_path = object_cls.file_path
obj_startPos = [0.5, 0, height]  # 比较下来发现0.67的高度刚好在桌上
obj_startOrientation = p.getQuaternionFromEuler([0, 0, 0.5*pi])
obj = object_init(obj_path, q_init=obj_startOrientation, t_init=obj_startPos, p=p)
obj_state = p.getBasePositionAndOrientation(obj.object_id)
# ====================== gpose prediction module initialization ======================== #
poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')

q_wo, t_wo = obj_startOrientation, obj_startPos
q_wo = object_rotation(q_wo)
print(len(poses))
target_gpose = poses[5]
target_gpose_wdc = gpose2wdc(target_gpose, q_wo, t_wo)

p.resetBasePositionAndOrientation(robot_id, target_gpose_wdc[4:], target_gpose_wdc[0:4])


p.setRealTimeSimulation(0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-90,
                                 cameraPitch=-37, cameraTargetPosition=obj_startPos)


while not keyboard.is_pressed('esc'):
    p.stepSimulation()
    time.sleep(1./240.)
    
# 断开连接
p.disconnect()