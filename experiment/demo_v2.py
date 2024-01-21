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


approach_time = 3  # 看到GO之后经过approach_time秒进入腕部运动
gap_time = 2 # wrist到grasp之间的时间
hold_time = 3

def init_pose(t=2.5,f=30):
    wrist_tf(f, 45)
    time.sleep(t)

def go_mug():
    # handle
    init_pose()
    print("GO MUG handle!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(0, -42)  # 抓把手的角度
    time.sleep(gap_time)  # 准备抓取
    grasp_handle()
    release_grasp(t=hold_time)  # 抓取持续时间
    # top
    init_pose(t=1, f=15)
    print("GO MUG top!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-30, 45)  # 抓top的角度
    time.sleep(gap_time)  # 准备抓取
    grasp_thin()
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def go_pottted_meat_can():
    # top
    init_pose()
    print("GO POTTED top!")
    time.sleep(approach_time)  # 靠近的时间
    wrist_tf(-15, 45)
    time.sleep(gap_time-1)  # 准备抓取
    # change mind
    # side
    print("GO POTTED side!")
    time.sleep(approach_time)  # 重新靠近的时间
    wrist_tf(0, -17)
    time.sleep(gap_time)  # 准备抓取
    grasp_other()
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def go_mustard_bottle():
    # side
    init_pose()
    print("GO MUSTARD side!")
    time.sleep(approach_time)  # 靠近的时间
    wrist_tf(-15, -25)
    time.sleep(gap_time)  # 准备抓取
    grasp_other()
    release_grasp(t=hold_time)  # 抓取持续时间
    # top
    init_pose(t=1, f=15)
    print("GO MUG top!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-30, 40)
    time.sleep(gap_time)  # 准备抓取
    grasp_other()
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def go_drill():
    # top
    init_pose()
    print("GO DRILL top!")
    time.sleep(approach_time)  # 靠近的时间
    wrist_tf(-30, 45)
    time.sleep(gap_time)  # 准备抓取
    grasp_medium()
    release_grasp(t=hold_time)  # 抓取持续时间
    # side
    init_pose(t=2, f=30)
    print("GO DRILL side!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-15, -45)
    time.sleep(gap_time)  # 准备抓取
    grasp_medium()
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)

def go_pitcher():
    # top
    init_pose()
    print("GO PITCHER top!")
    time.sleep(approach_time)  # 靠近的时间
    wrist_tf(-30, 45)
    time.sleep(gap_time)  # 准备抓取
    grasp_medium()
    release_grasp(t=hold_time+1.5)  # 抓取持续时间
    # side
    init_pose(t=2, f=30)
    print("GO PITCHER side!")
    time.sleep(approach_time)  #靠近的时间
    wrist_tf(-10, -45)
    time.sleep(gap_time)  # 准备抓取
    grasp_pitcher()
    time.sleep(2.5)
    print("put down~")
    release_grasp(t=hold_time)  # 抓取持续时间
    # end
    init_pose(t=1.5)




if __name__ == "__main__":
    wrist_tf(30, 45)
    while True:
        if keyboard.is_pressed('space'):
            print("preparing!")
            go_pitcher()
            go_mug()
            go_pottted_meat_can()
            go_mustard_bottle()
            go_drill()
        if keyboard.is_pressed('esc'):
            print("end")
            break
    wrist_tf(30, 45)
    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 
    

    
    

    
    