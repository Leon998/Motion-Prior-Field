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

# =================================== pitcher ================================ #
def mpf_pitcher():
    # side
    init_pose()
    print("GO PITCHER side!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-10, -45)
    time.sleep(gap_time)  # 准备抓取
    grasp_pitcher()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def tro_pitcher():
    # side
    init_pose()
    print("GO PITCHER side!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-20, 30)  # 错误姿势
    time.sleep(3.5)
    wrist_tf(-20, -40)  # 正确姿势
    time.sleep(2.5)
    grasp_other()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def emg_pitcher():
    # side
    init_pose()
    print("GO PITCHER side!")
    time.sleep(1)
    wrist_tf(30, 15)  # 转第一次
    time.sleep(1)
    wrist_tf(30, -15)  # 转第二次
    time.sleep(1.5)
    wrist_tf(30, -45)  # 转第三次
    time.sleep(1.5)
    wrist_tf(30, -30)  # 转错
    time.sleep(1)
    wrist_tf(10, -30)  # 最后一次
    time.sleep(2.5)
    grasp_medium()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)



# ================================== drill ==================================== #
def mpf_drill():
    # side
    init_pose()
    print("GO DRILL side!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-20, -45)
    time.sleep(gap_time)  # 准备抓取
    grasp_medium()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def tro_drill():
    # side
    init_pose()
    print("GO DRILL side!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(25, -45)
    time.sleep(gap_time)  # 准备抓取
    grasp_medium()
    time.sleep(hold_time)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def emg_drill():
    # side
    init_pose()
    print("GO DRILL side!")
    time.sleep(1)
    wrist_tf(30, 0)  # 转第一次
    time.sleep(1.5)
    wrist_tf(30, -45)  # 转第二次
    time.sleep(1.5)
    wrist_tf(30, -30)  # 转错
    time.sleep(1)
    wrist_tf(30, -45)  # 转错
    time.sleep(1)
    wrist_tf(-10, -45)  # 最后一次
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
            emg_drill()
            break
        if keyboard.is_pressed('esc'):
            print("end")
            break
    wrist_tf(30, 45)
    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 
    

    
    

    
    