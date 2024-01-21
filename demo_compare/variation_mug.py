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
hold_time = 1.5

def init_pose(t=2.5,f=30):
    wrist_tf(f, 45)
    time.sleep(t)

def mpf_mug():
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-25, -42)  # 抓把手的角度
    time.sleep(gap_time)  # 准备抓取
    grasp_handle()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)
    # ===================== 转移 =================== #
    time.sleep(3)
    print("Over!")
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(30, -45)  # 抓把手的角度
    time.sleep(gap_time)  # 准备抓取
    grasp_handle()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def tro_mug():
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(approach_time+0.5)  #靠近的时间
    wrist_tf(5, -45)  # 抓把手的角度
    time.sleep(gap_time+0.3)  # 准备抓取
    grasp_other()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)
    # ===================== 转移 =================== #
    time.sleep(3)
    print("Over!")
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(10, -45)  # 抓把手的角度
    time.sleep(gap_time+0.5)  # 准备抓取
    grasp_other()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def emg_mug():
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(1)
    wrist_tf(30, -15)  # 转第一次
    time.sleep(2)
    wrist_tf(30, -45)  # 转第二次
    time.sleep(1.5)
    wrist_tf(30, -30)  # 转错
    time.sleep(1)
    wrist_tf(30, -45)  # 转对
    time.sleep(1)
    wrist_tf(-30, -30)  # 最后一次
    time.sleep(2.5)
    grasp_thin()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)
    # ===================== 转移 =================== #
    time.sleep(3)
    print("Over!")
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(1)
    wrist_tf(30, -45)  # 转第一次
    time.sleep(2.5)
    wrist_tf(30, -30)  # 抓取识别成转腕
    time.sleep(2.5)
    grasp_medium()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)


if __name__ == "__main__":
    wrist_tf(30, 45)
    while True:
        if keyboard.is_pressed('space'):
            print("preparing!")
            emg_mug()
            break
        if keyboard.is_pressed('esc'):
            print("end")
            break
    wrist_tf(30, 45)
    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 
    

    
    

    
    