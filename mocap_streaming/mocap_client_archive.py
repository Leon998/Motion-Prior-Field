import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from myutils.utils import *
import open3d as o3d
import redis

# pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Visualize
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='vis', width=1080, height=720)

object_cls = objects['pitcher_base']
object_mesh = object_cls.init_transform()
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
init_hand = load_mano()
i = 0
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
    # print(hand_pose)
    translation = tuple(hand_pose[4:])
    print("translation:",translation)
    current_hand = hand_transform(hand_pose, init_hand)

    # ========================= visualize in open3d ==================== #
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('simulation/BV_1440.json')
    
    vis.clear_geometries()
    vis.add_geometry(coordinate)
    vis.add_geometry(current_hand)
    vis.add_geometry(object_mesh)
    print(current_hand.get_center() - object_mesh.get_center())

    vis.get_render_option().load_from_json('simulation/renderoption.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    # 更新窗口
    vis.poll_events()
    vis.update_renderer()
    i += 1
    # if i > 1:
    #     break


    