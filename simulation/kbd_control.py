import pybullet as p
import pybullet_data
from time import sleep
from math import pi

use_gui = True
if use_gui:
    cid = p.connect(p.GUI)
else:
    cid = p.connect(p.DIRECT)

p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
# plane_id = p.loadURDF("plane.urdf", useMaximalCoordinates=False)
robot_path = "simulation/wrist_hand_left_v2/urdf/wrist_hand_left_v2.urdf"
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_path, startPos, startOrientation, useFixedBase=1)


p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135,
                                 cameraPitch=-30, cameraTargetPosition=[0,0,0])
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(1)

maxV = 2
maxF = 0.01

while True:
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

p.disconnect(cid)