import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import objects
from myutils.utils import *
import open3d as o3d
import torch
sys.path.append('pose_estimation/')
from pose_estimation.utils import *
from pose_estimation.utils4coordinate_tf import *
import redis
from can.wrist_control import *
import keyboard


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
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [coordinate, object_mesh]
    device = "cuda"

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis', width=720, height=640)
    vis.add_geometry(coordinate)
    vis.add_geometry(object_mesh)
    hand = load_mano()
    pred_hand = load_mano()
    pred_hand.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
    vis.add_geometry(hand)
    vis.add_geometry(pred_hand)
    translation = (0, 0, 0)
    rotation = (1, 0, 0, 0)
    pred_hand_translation = (0, 0, 0)
    pred_hand_rotation = (1, 0, 0, 0)
    wrist_rotation = 0
    wrist_flexion = 0

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
        # object
        t_wo = np.array([float(i) for i in r.get('object_position')[1:-1].split(',')])
        q_wo = np.array([float(i) for i in r.get('object_rotation')[1:-1].split(',')])

        q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
        hand_pose = np.concatenate((q_oh, t_oh), axis=0)

        translation, rotation, delta_translation, delta_R = incremental_hand_transform(hand, 
                                                                                   hand_pose, 
                                                                                   translation, 
                                                                                   rotation)

        hand.translate(delta_translation, relative=True)
        hand.rotate(delta_R, center=translation)

        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        pred_gpose = poses[idx]

        # pred_hand transform
        pred_hand_translation, pred_hand_rotation, pred_hand_delta_translation, pred_hand_delta_R = incremental_hand_transform(pred_hand, 
                                                                                   pred_gpose, 
                                                                                   pred_hand_translation, 
                                                                                   pred_hand_rotation)

        pred_hand.translate(pred_hand_delta_translation, relative=True)
        pred_hand.rotate(pred_hand_delta_R, center=pred_hand_translation)

        # ======================== wrist joint transformation ============================= #
        euler_joint, r_transform = wrist_joint_transform(hand_pose, pred_gpose)
        # print(euler_joint)
        
        

        # transformed_hand = copy.deepcopy(hand)
        # transformed_hand.rotate(r_transform, center=hand_pose[4:])
        # transformed_hand.paint_uniform_color([255 / 255, 190 / 255, 122 / 255])

        # ========================= visualize in open3d ==================== #
        vis.update_geometry(hand)
        vis.update_geometry(pred_hand)

        # 更新窗口
        vis.poll_events()
        vis.update_renderer()

        # ============================ Wrist control ======================= #
        if keyboard.is_pressed('enter'):
            print("current joint angle = ", wrist_rotation, wrist_flexion)
            print("current joint trans = ", - euler_joint[2], euler_joint[0])
            wrist_rotation = - euler_joint[2]
            wrist_flexion = euler_joint[0]
            print("updated joint angle = ", wrist_rotation, wrist_flexion)
            print("\n")
            
            # 转十六进制
            hex_wrist_rotation = int(wrist_rotation * 255 / 360) + 128
            hex_wrist_flexion = int(wrist_flexion * 255 / 360) + 128
            a = ubyte_array(0, hex_wrist_rotation, hex_wrist_flexion, 0, 0, 0, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
            ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
            time.sleep(1)
        elif keyboard.is_pressed('q'):
            a = ubyte_array(0, 128, 128, 0, 0, 0, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
            ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)


