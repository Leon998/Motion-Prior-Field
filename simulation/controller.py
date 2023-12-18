import pybullet as p
import time
import pybullet_data
from math import pi


def auto_controller(robot_id, p, joint_positions):
    maxV = [2, 2, 2]
    maxF = [0.5, 0.5, 0.5]
    p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[0, 1, 2],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions,
            targetVelocities=maxV,
            forces=maxF
            )

def kbd_controller(robot_id, p):
    maxV = 5
    maxF = 0.2
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


if __name__ == "__main__":
    # 连接物理引擎
    physicsCilent = p.connect(p.GUI)
    # 渲染逻辑
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # 添加资源路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 设置环境重力加速度
    p.setGravity(0, 0, 0)
    # 加载URDF模型，此处是加载蓝白相间的陆地
    planeId = p.loadURDF("plane.urdf")
    # 加载机器人，并设置加载的机器人的位姿
    startPos = [0, 0, 0.5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Mocap坐标系
    robot_id = p.loadURDF("simulation/wrist_hand_left_v2/urdf/wrist_hand_left_v2.urdf", startPos, startOrientation, useFixedBase=1)
    joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135,
                                     cameraPitch=-30, cameraTargetPosition=[0,0,0.5])

    target_positions = [0.209124569199306, -0.17839500851555457, 1.9202608323759964]
    i = 0
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1./240.)
        if i == 50:
            auto_controller(robot_id, p, target_positions)
        if i > 75:
            kbd_controller(robot_id, p)


    # 断开连接
    p.disconnect()