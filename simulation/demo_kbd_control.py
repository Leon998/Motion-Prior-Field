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
import redis
from can.archive_wrist_control import *
import keyboard


def auto_control(degree):
    ubyte_array = c_ubyte*8
    ubyte_3array = c_ubyte*3
    flexion_degree = degree[0]
    rotation_degree = degree[1]
    wrist_rotation = rotation_degree if rotation_degree >= 0 else -rotation_degree + 128
    wrist_flexion = flexion_degree if flexion_degree >= 0 else -flexion_degree + 128
    a = ubyte_array(2, 0, wrist_flexion, wrist_rotation, 0, 0, 0, 0)
    b = ubyte_3array(0, 0 , 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141314, 0, 0, 1, 0, 1,  8, a, b)#单次发送，0x14为手腕id
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)


if __name__ == "__main__":
    ubyte_array = c_ubyte*8
    ubyte_3array = c_ubyte*3
    while True:
        # ============================ Wrist control ======================= #
        # if keyboard.is_pressed('0'):
        #     degree = (-45, -5)
        #     flexion_degree = degree[0]
        #     rotation_degree = degree[1]
        #     wrist_rotation = rotation_degree if rotation_degree >= 0 else -rotation_degree + 128
        #     wrist_flexion = flexion_degree if flexion_degree >= 0 else -flexion_degree + 128
        #     a = ubyte_array(2, 0, wrist_flexion, wrist_rotation, 0, 0, 0, 0)
        #     b = ubyte_3array(0, 0 , 0)
        #     vci_can_obj = VCI_CAN_OBJ(0x13141314, 0, 0, 1, 0, 1,  8, a, b)#单次发送，0x14为手腕id
        #     ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
        #     time.sleep(1)
        # elif keyboard.is_pressed('1'):
        #     degree = (-10, 40)
        #     flexion_degree = degree[0]
        #     rotation_degree = degree[1]
        #     wrist_rotation = rotation_degree if rotation_degree >= 0 else -rotation_degree + 128
        #     wrist_flexion = flexion_degree if flexion_degree >= 0 else -flexion_degree + 128
        #     a = ubyte_array(2, 0, wrist_flexion, wrist_rotation, 0, 0, 0, 0)
        #     b = ubyte_3array(0, 0 , 0)
        #     vci_can_obj = VCI_CAN_OBJ(0x13141314, 0, 0, 1, 0, 1,  8, a, b)#单次发送，0x14为手腕id
        #     ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
        #     time.sleep(1)
        # elif keyboard.is_pressed('2'):
        #     degree = (45, -45)
        #     flexion_degree = degree[0]
        #     rotation_degree = degree[1]
        #     wrist_rotation = rotation_degree if rotation_degree >= 0 else -rotation_degree + 128
        #     wrist_flexion = flexion_degree if flexion_degree >= 0 else -flexion_degree + 128
        #     a = ubyte_array(2, 0, wrist_flexion, wrist_rotation, 0, 0, 0, 0)
        #     b = ubyte_3array(0, 0 , 0)
        #     vci_can_obj = VCI_CAN_OBJ(0x13141314, 0, 0, 1, 0, 1,  8, a, b)#单次发送，0x14为手腕id
        #     ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
        #     time.sleep(1)
        degree = (0, 0)
        auto_control(degree)
        time.sleep(3)

        degree = (30, 80)
        auto_control(degree)
        time.sleep(3)

        degree = (50, 20)
        auto_control(degree)
        time.sleep(3)
        degree = (0, 0)
        auto_control(degree)

        break