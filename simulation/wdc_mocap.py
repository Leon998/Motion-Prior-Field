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
from can.hand_control import *
import keyboard, time
import math, random


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
    vis.add_geometry(hand.mesh)

    target_idx = random.randint(0,len(poses))
    if object_cls == mug & target_idx < 60:
        grasp_type = grasp_handle
    else:
        grasp_type = grasp_other
    target_gpose = poses[target_idx]
    target_hand = assets(mesh=load_mano())
    target_hand.mesh.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
    vis.add_geometry(target_hand.mesh)

    pred_hand = assets(mesh=load_mano())
    pred_hand.mesh.paint_uniform_color([250 / 255, 127 / 255, 111 / 255])
    vis.add_geometry(pred_hand.mesh)

    # 初始化手腕位置
    ubyte_array = c_ubyte*8
    a = ubyte_array(0, 128, 128, 0, 0, 0, 0, 0)
    ubyte_3array = c_ubyte*3
    b = ubyte_3array(0, 0 , 0)
    vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
 
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)

    # flexion_degree, rotation_degree = read_wrist()
    flexion_degree, rotation_degree = 0, 0

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
        target_gpose_wdc = gpose2wdc(target_gpose, q_wo, t_wo)
        
        # ======================== grasp pose prediction ============================= #
        x = torch.from_numpy(hand_pose).type(torch.FloatTensor).to(device)
        pred = model(x)
        idx = pred.argmax(0).item()
        pred_gpose = poses[idx]
        pred_gpose_wdc = gpose2wdc(pred_gpose, q_wo, t_wo)

        # ========================= update transform ==================== #
        hand.update_transform(hand_pose_wdc)
        target_hand.update_transform(target_gpose_wdc)
        pred_hand.update_transform(pred_gpose_wdc)
        object.update_transform(object_pose_wdc)

        vis.update_geometry(hand.mesh)
        vis.update_geometry(target_hand.mesh)
        vis.update_geometry(pred_hand.mesh)
        vis.update_geometry(object.mesh)

        # 更新窗口
        p1, p2 = hand_pose_wdc[-3:], object_pose_wdc[-3:]
        p3=p2-p1
        p4=math.sqrt(pow(p3[0],2)+pow(p3[1],2)+pow(p3[2],2))
        if p4 > 0.2:
            # 距离足够远才更新
            vis.poll_events()
            vis.update_renderer()

        # ======================== wrist joint transformation ============================= #
        if keyboard.is_pressed('ctrl'):
            euler_joint, r_transform = wrist_joint_transform(hand_pose, pred_gpose)
            flexion_degree += -euler_joint[0]
            rotation_degree += -euler_joint[2]
            flexion_degree, rotation_degree = wrist_limit(flexion_degree, rotation_degree)
            wrist_tf(flexion_degree, rotation_degree)
            hand_tf(0xA1, 0x02)
            time.sleep(1.5)
            flexion_degree, rotation_degree = read_wrist()
        elif keyboard.is_pressed('backspace'):
            wrist_tf(0, 0)
            time.sleep(1.5)
            flexion_degree, rotation_degree = read_wrist()
        elif keyboard.is_pressed('enter'):
            grasp_type()
        elif keyboard.is_pressed('esc'):
            break
        
        print(flexion_degree, rotation_degree)

    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 



