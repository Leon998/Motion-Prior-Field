import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import load_mano
from myutils.object_config import objects
from myutils.assets_config import assets
from myutils.utils import *
import open3d as o3d
import torch
import redis
from can.archive_wrist_control import *
import keyboard, time


if __name__ == "__main__":  
    # pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)  
    # ====================== gpose prediction module initialization ======================== #
    object_cls = objects['mug']
    poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
    model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')
    model.eval()

    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    device = "cuda"

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis', width=1080, height=720)
    vis.add_geometry(coordinate)

    object = assets(mesh=object_cls.init_transform())
    vis.add_geometry(object.mesh)

    hand = assets(mesh=load_mano())
    pred_hand = assets(mesh=load_mano())
    pred_hand.mesh.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
    vis.add_geometry(hand.mesh)
    vis.add_geometry(pred_hand.mesh)

    # 初始化手腕位置
    ubyte_array = c_ubyte*8
    a = ubyte_array(0, 128, 128, 0, 0, 0, 0, 0)
    ubyte_3array = c_ubyte*3
    b = ubyte_3array(0, 0 , 0)
    vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
 
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)

    while True:
        # hand
        t_wh = np.array([float(i) for i in r.get('hand_position')[1:-1].split(',')])
        q_wh = np.array([float(i) for i in r.get('hand_rotation')[1:-1].split(',')])
        hand_pose_wdc = np.concatenate((q_wh, t_wh), axis=0)
        # object
        t_wo = np.array([float(i) for i in r.get('object_position')[1:-1].split(',')])
        q_wo = np.array([float(i) for i in r.get('object_rotation')[1:-1].split(',')])
        object_pose_wdc = np.concatenate((q_wo, t_wo))

        q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
        hand_pose = np.concatenate((q_oh, t_oh), axis=0)
        
        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        pred_gpose = poses[idx]
        pred_gpose_wdc = gpose2wdc(pred_gpose, q_wo, t_wo)

        # ======================== wrist joint transformation ============================= #
        euler_joint, r_transform = wrist_joint_transform(hand_pose, pred_gpose)
        # print(euler_joint)
        

        # ========================= update transform ==================== #
        hand.update_transform(hand_pose_wdc)
        pred_hand.update_transform(pred_gpose_wdc)
        object.update_transform(object_pose_wdc)

        vis.update_geometry(hand.mesh)
        vis.update_geometry(pred_hand.mesh)
        vis.update_geometry(object.mesh)

        # 更新窗口
        vis.poll_events()
        vis.update_renderer()

        # ============================ Wrist control ======================= #
        # if keyboard.is_pressed('enter'):
        #     print("current joint angle = ", wrist_rotation, wrist_flexion)
        #     print("current joint trans = ", - euler_joint[2], euler_joint[0])
        #     # 如果只要开环控制的话，这里就写一个判定函数，来记录和变更关节角。
        #     wrist_rotation = wrist_rotation - euler_joint[2]
        #     if wrist_rotation < -100:
        #         wrist_rotation = -100
        #     elif wrist_rotation > 100:
        #         wrist_rotation = 100
        #     wrist_flexion = wrist_flexion + euler_joint[0]
        #     if wrist_flexion < -45:
        #         wrist_flexion = -45
        #     elif wrist_flexion > 45:
        #         wrist_flexion = 45

        #     print("updated joint angle = ", wrist_rotation, wrist_flexion)
        #     print("\n")
            
        #     # 转十六进制
        #     hex_wrist_rotation = int(wrist_rotation * 255 / 360) + 128
        #     hex_wrist_flexion = int(wrist_flexion * 255 / 360) + 128
        #     a = ubyte_array(0, hex_wrist_rotation, hex_wrist_flexion, 0, 0, 0, 0, 0)
        #     vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
        #     ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
        #     time.sleep(1)
        # elif keyboard.is_pressed('q'):
        #     a = ubyte_array(0, 128, 128, 0, 0, 0, 0, 0)
        #     vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
        #     ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)


