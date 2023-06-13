import os
import sys
sys.path.append(os.getcwd())
from myutils.utils import *
import numpy as np
from myutils.object_config import objects, colorlib
from myutils.hand_config import *
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from can.wrist_control import *
import keyboard

if __name__ == "__main__":
    object_cls = objects['mug']
    path = 'mocap/' + object_cls.name + '/'
    # Source files
    source_files = os.listdir(path)
    source_files.sort()
    idx = 150 # 随便选的一个抓取的序号
    file = source_files[idx]
    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [object_mesh]

    file_path = path + file
    Q_wh, T_wh, Q_wo, T_wo, num_frame = read_data(file_path)
    Q_oh, T_oh, TF_oh = sequence_coordinate_transform(Q_wh, T_wh, Q_wo, T_wo, num_frame)

    # ==================================================== #

    # Grasp pose
    gpose = TF_oh[-1, :]
    g_hand = hand_transform(gpose, init_hand)
    g_hand.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
    meshes.append(g_hand)

    # Mid pose
    length = TF_oh.shape[0]
    mid_time = int(0.3 * length)
    mid_pose = TF_oh[mid_time]
    mid_hand = hand_transform(mid_pose, init_hand)
    meshes.append(mid_hand)

    # Target pose according to grasp pose
    # target_pose = np.concatenate((gpose[:4], mid_pose[4:]))
    # target_hand = hand_transform(target_pose, init_hand)
    # target_hand.paint_uniform_color([142 / 255, 207 / 255, 201 / 255])
    # meshes.append(target_hand)

    print(mid_pose)
    # print(target_pose)

    # ========================================================== #
    # Transform
    r_mid_pose = R.from_quat(mid_pose[:4]).as_matrix()
    r_target_pose = R.from_quat(gpose[:4]).as_matrix()
    r_transform = (r_target_pose).dot(np.linalg.inv(r_mid_pose))
    # q_transform = R.from_matrix(r_transform).as_quat()
    euler_transform = R.from_matrix(r_transform).as_euler('zyx', degrees=True)
    print(euler_transform)
    wrist_rotate = euler_transform[0]
    wrist_flip = euler_transform[2]

    transformed_hand = copy.deepcopy(mid_hand)
    transformed_hand.rotate(r_transform, center=mid_pose[4:])
    transformed_hand.paint_uniform_color([255 / 255, 190 / 255, 122 / 255])
    meshes.append(transformed_hand)
    
    o3d.visualization.draw_geometries(meshes)

    # ============================ Wrist control ======================= #
    ubyte_array = c_ubyte*8
    wrist_rotate = - int(wrist_rotate * 255 / 180) + 128
    wrist_flip = - int(wrist_flip * 255 / 90) + 128
    a = ubyte_array(0, wrist_rotate, wrist_flip, 0, 0, 0, 0, 0)
    ubyte_3array = c_ubyte*3
    b = ubyte_3array(0, 0 , 0)
    vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id
    
    while True:
        if keyboard.is_pressed('enter'):
            ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
            #关闭
            canDLL.VCI_CloseDevice(VCI_USBCAN2, 0)
            break

    
    
