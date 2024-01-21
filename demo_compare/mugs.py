# 总demo

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
import time


approach_time = 2  # 看到GO之后经过approach_time秒进入腕部运动
gap_time = 2.5 # wrist到grasp之间的时间
hold_time = 2

def init_pose(t=2.5,f=30):
    wrist_tf(f, 45)
    time.sleep(t)

def grasp_mugs(theta1, theta2):
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(theta1, theta2)  # 抓把手的角度
    time.sleep(gap_time)  # 准备抓取
    grasp_handle()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=3)




if __name__ == "__main__":
    wrist_tf(30, 45)
    while True:
        if keyboard.is_pressed('space'):
            print("preparing!")
            grasp_mugs(-25, -42)
            grasp_mugs(30, -45)
            grasp_mugs(0, -40)
            grasp_mugs(-20, -45)
            break
        if keyboard.is_pressed('esc'):
            print("end")
            break
    wrist_tf(30, 45)
    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 
    

    
    

    
    