# 方法对比demo

import os
import sys
sys.path.append(os.getcwd())
from ctypes import *
import time, keyboard
import numpy as np
from can.hand_control import *
import redis


flexion_degree, rotation_degree = 0, 0
d_flexion = 5
d_rotation = 10

while True:
    flexion_degree, rotation_degree = read_wrist()
    if keyboard.is_pressed('left'):
        d_tf = wrist_limit(flexion_degree, rotation_degree-d_rotation)
        wrist_tf(d_tf[0], d_tf[1])
    elif keyboard.is_pressed('right'):
        d_tf = wrist_limit(flexion_degree, rotation_degree+d_rotation)
        wrist_tf(d_tf[0], d_tf[1])
    elif keyboard.is_pressed('up'):
        d_tf = wrist_limit(flexion_degree+d_flexion, rotation_degree)
        wrist_tf(d_tf[0], d_tf[1])
    elif keyboard.is_pressed('down'):
        d_tf = wrist_limit(flexion_degree-d_flexion, rotation_degree)
        wrist_tf(d_tf[0], d_tf[1])
    elif keyboard.is_pressed('1'):
        grasp_handle()
    elif keyboard.is_pressed('2'):
        grasp_thin()
    elif keyboard.is_pressed('3'):
        grasp_medium()
    elif keyboard.is_pressed('4'):
        grasp_other()
    elif keyboard.is_pressed('5'):
        grasp_pitcher()
    elif keyboard.is_pressed('7'):
        hand_tf(0xA1, 0x02)
    if keyboard.is_pressed('space'):
        # release_grasp()
        hand_tf(0xA1, 0x01)
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