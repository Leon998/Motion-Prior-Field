import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from myutils.utils import *
import open3d as o3d
import redis
import time

# pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Visualize
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='vis')

object_cls = objects['mug']
object_mesh = object_cls.init_transform()
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

vis.add_geometry(coordinate)
vis.add_geometry(object_mesh)
hand = load_mano()
hand_bias = hand.get_center()
vis.add_geometry(hand)
rotation = (1, 0, 0, 0)
translation = (0, 0, 0)
while True:
    # hand
    t_wh = np.array([float(i) for i in r.get('hand_position')[1:-1].split(',')])
    q_wh = np.array([float(i) for i in r.get('hand_rotation')[1:-1].split(',')])
    # object
    t_wo = np.array([float(i) for i in r.get('object_position')[1:-1].split(',')])
    q_wo = np.array([float(i) for i in r.get('object_rotation')[1:-1].split(',')])
    # ========================= Object coordicate ==================== #
    q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
    hand_pose = np.concatenate((q_oh, t_oh), axis=0)

    # hand transform
    # rotation, translation, R_transform = incremental_hand_transform(hand, hand_pose, rotation)
    last_translation = np.array(translation)
    translation = np.array(hand_pose[4:])
    delta_translation = translation - last_translation
    print("delta_translation:", delta_translation) 
    hand.translate(delta_translation, relative=True)
    
    last_rotation = rotation
    rotation = tuple((hand_pose[3], hand_pose[0], hand_pose[1], hand_pose[2]))
    last_R = hand.get_rotation_matrix_from_quaternion(last_rotation)
    current_R = hand.get_rotation_matrix_from_quaternion(rotation)
    delta_R = (current_R).dot(np.linalg.inv(last_R))
    # delta_R = (np.linalg.inv(current_R)).dot(last_R)  
    hand.rotate(delta_R, center=translation)

    # ========================= visualize in open3d ==================== #
    vis.update_geometry(hand)
    vis.poll_events()
    vis.update_renderer()


    