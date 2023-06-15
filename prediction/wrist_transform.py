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
    idx = 50  # 随便选的一个抓取的序号
    file = source_files[idx]
    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [coordinate, object_mesh]

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
    mid_time = int(0.4 * length)
    mid_pose = TF_oh[mid_time]
    mid_hand = hand_transform(mid_pose, init_hand)
    meshes.append(mid_hand)

    print("mid_pose=", mid_pose)
    print("target_pose=", gpose[:4])

    # ========================================================== #
    # Transform
    r_mid_pose = R.from_quat(mid_pose[:4]).as_matrix()
    r_target_pose = R.from_quat(gpose[:4]).as_matrix()
    r_transform = (r_target_pose).dot(np.linalg.inv(r_mid_pose))  # 用来可视化变换
    r_joint = np.linalg.inv(r_mid_pose).dot(r_target_pose)  # 用来计算关节变换角

    # 以下代码有问题，正在排查
    # 这里是将mid_hand到target_pose之间的变换矩阵r_transform直接输出成欧拉角了，因此代表变换矩阵
    # 而我们控制假肢手时，相当于把手从初始位置变换到target_pose，
    # 因此可以考虑将target_hand和mid_hand都乘一个mid_pose的逆，变换到init_hand，在来反解旋转角
    euler_joint = R.from_matrix(r_joint).as_euler('zyx', degrees=True)
    print(euler_joint)
    wrist_rotation = - euler_joint[2]
    wrist_flexion = euler_joint[0]
    
    
    
    transformed_hand = copy.deepcopy(mid_hand)
    transformed_hand.rotate(r_transform, center=mid_pose[4:])

    transformed_hand.paint_uniform_color([255 / 255, 190 / 255, 122 / 255])
    meshes.append(transformed_hand)
    
    o3d.visualization.draw_geometries(meshes)

    # ============================ Wrist control ======================= #
    ubyte_array = c_ubyte*8
    
    hex_wrist_rotation = int(wrist_rotation * 255 / 360) + 128
    hex_wrist_flexion = int(wrist_flexion * 255 / 360) + 128
    a = ubyte_array(0, hex_wrist_rotation, hex_wrist_flexion, 0, 0, 0, 0, 0)
    ubyte_3array = c_ubyte*3
    b = ubyte_3array(0, 0 , 0)
    vci_can_obj = VCI_CAN_OBJ(0x14, 0, 0, 1, 0, 0,  8, a, b)#单次发送，0x14为手腕id

    # ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    # #关闭
    # canDLL.VCI_CloseDevice(VCI_USBCAN2, 0)

    while True:
        if keyboard.is_pressed('enter'):
            print("go")
            ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
            #关闭
            canDLL.VCI_CloseDevice(VCI_USBCAN2, 0)
            break


    
    
