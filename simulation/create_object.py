import pybullet as p
import pybullet_data
import time
from math import pi

class object_init:
    def __init__(self, 
                 file_path, 
                 q_init: object=(0, 0, 0, 1),
                 t_init: object=(0, 0, 0),
                 p: object=0):
        self.file_path = file_path
        self.q_init = q_init
        self.t_init = t_init
        self.object_id = 0
        # 创建视觉形状和碰撞箱形状
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=self.file_path,
            rgbaColor=[0.9, 0, 0, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1],
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self.file_path,
            collisionFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1]
        )
        # 使用创建的视觉形状和碰撞箱形状使用createMultiBody将两者结合在一起
        self.object_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            useMaximalCoordinates=True
        )

        p.resetBasePositionAndOrientation(self.object_id, self.t_init, self.q_init)




if __name__ == "__main__":
    _ = p.connect(p.GUI)
    obj_startPos = [0, 0, 0]
    obj_startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    obj = object_init("models/025_mug/textured_simple.obj", q_init=obj_startOrientation, t_init=obj_startPos, p=p)
    # 添加资源路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=10)

    # 创建过程中不渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # 不展示GUI的套件
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    # 禁用 tinyrenderer 
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)
    # 创建结束，重新开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135,
                                     cameraPitch=-30, cameraTargetPosition=[0,0,0])
    while p.isConnected():
        time.sleep(1./240.)