import os
import sys
sys.path.append(os.getcwd())
from ctypes import *
import time, keyboard
import numpy as np
from can.hand_control import *
import redis
from collections import deque


action = deque(maxlen=5)
grasp_action = deque(maxlen=8)


if __name__ == "__main__":
    rds = redis.Redis(host='localhost', port=6379, decode_responses=True)
    flexion_degree, rotation_degree = 0, 0
    d_flexion = 5
    d_rotation = 10
    grasp = grasp_handle
    action = deque(maxlen=5)
    grasp_action = deque(maxlen=8)
    while True:
        action.append(int(rds.get('action')))
        grasp_action.append(int(rds.get('action')))
        flexion_degree, rotation_degree = read_wrist()
        if all(x == 0 for x in action):
            pass
        elif all(x == 1 for x in action):
            d_tf = wrist_limit(flexion_degree+d_flexion, rotation_degree)
            wrist_tf(d_tf[0], d_tf[1])
        elif all(x == 2 for x in action):
            d_tf = wrist_limit(flexion_degree-d_flexion, rotation_degree)
            wrist_tf(d_tf[0], d_tf[1])
        elif all(x == 3 for x in action):
            d_tf = wrist_limit(flexion_degree, rotation_degree-d_rotation)
            wrist_tf(d_tf[0], d_tf[1])
        elif all(x == 4 for x in action):
            d_tf = wrist_limit(flexion_degree, rotation_degree+d_rotation)
            wrist_tf(d_tf[0], d_tf[1])
        elif all(x == 5 for x in grasp_action) or keyboard.is_pressed('enter'):
            print("Grasping!")
            grasp()
            time.sleep(1.5)
            release_grasp()
        # ============================= kbd control part ======================= #
        if keyboard.is_pressed('backspace'):
            print("reset pose")
            release_grasp()
            wrist_tf(20, 45)
            time.sleep(1.5)
            flexion_degree, rotation_degree = read_wrist()
            # print(flexion_degree, rotation_degree)

        if keyboard.is_pressed('esc'):
            print("END")
            break
        
        print(flexion_degree, rotation_degree)


    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0)