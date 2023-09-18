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



def release_grasp_long(t=2.5):
    time.sleep(t)
    hand_tf(0xA1, 0x01)

def init_pose(t=5,f=30):
    wrist_tf(f, -45)
    time.sleep(t)

if __name__ == "__main__":
    # ======= DO NOT CHANGE ========== #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    time.sleep(3)
    init_pose()
    # mug handle
    wrist_tf(0, 40)
    time.sleep(1.5)
    grasp_handle()
    release_grasp_long()
    # mug top
    init_pose(t=3.5,f=0)
    wrist_tf(-30, -45)
    time.sleep(1.5)
    grasp_mug_top()
    release_grasp_long()

    init_pose()
    # potted_meat_can top
    wrist_tf(-15, -40)
    time.sleep(1.5)
    # change mind
    # potted_meat_can side
    time.sleep(2.5)
    wrist_tf(0, 20)
    time.sleep(1.5)
    grasp_other()
    release_grasp_long()
    

    init_pose()
    # mustard_bottle side
    wrist_tf(0, 20)
    time.sleep(1.5)
    grasp_other()
    release_grasp_long()
    # mustard_bottle top
    init_pose(t=3.5,f=0)
    wrist_tf(-30, -40)
    time.sleep(1.5)
    grasp_other()
    release_grasp_long()
    init_pose(t=2)
    