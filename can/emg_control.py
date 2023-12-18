from ctypes import *
import time, keyboard
import numpy as np
from hand_control import *
import redis
 

if __name__ == "__main__":
    rds = redis.Redis(host='localhost', port=6379, decode_responses=True)
    flexion_degree, rotation_degree = 0, 0
    d_flexion = 5
    d_rotation = 10
    grasp_type = grasp_handle
    while True:
        action = int(rds.get('action'))
        flexion_degree, rotation_degree = read_wrist()
        if action == 0:
            pass
        elif action == 1:
            d_tf = wrist_limit(flexion_degree+d_flexion, rotation_degree)
            wrist_tf(d_tf[0], d_tf[1])
        elif action == 2:
            d_tf = wrist_limit(flexion_degree-d_flexion, rotation_degree)
            wrist_tf(d_tf[0], d_tf[1])
        elif action == 3:
            d_tf = wrist_limit(flexion_degree, rotation_degree-d_rotation)
            wrist_tf(d_tf[0], d_tf[1])
        elif action == 4:
            d_tf = wrist_limit(flexion_degree, rotation_degree+d_rotation)
            wrist_tf(d_tf[0], d_tf[1])
        elif action == 5 or keyboard.is_pressed('enter'):
            print("Grasping!")
            grasp()
            time.sleep(1.5)
            release_grasp()
        # ============================= kbd control part ======================= #
        elif keyboard.is_pressed('backspace'):
            print("reset pose")
            release_grasp()
            wrist_tf(0, 45)
            time.sleep(1.5)
            flexion_degree, rotation_degree = read_wrist()
            # print(flexion_degree, rotation_degree)

        elif keyboard.is_pressed('esc'):
            print("END")
            break
        
        print(flexion_degree, rotation_degree)


    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0)