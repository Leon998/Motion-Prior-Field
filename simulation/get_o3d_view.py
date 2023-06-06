import os
import numpy as np
import struct
import open3d as o3d
import time
import os
import sys
sys.path.append(os.getcwd())
from myutils.object_config import objects

from open3d import visualization


def read_bin_velodyne(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def save_view_point(pcd, filename):
    vis = visualization.Visualizer()
    vis.create_window(window_name='vis', width=1080, height=720)
    vis.add_geometry(pcd)

    # vis.get_render_option().load_from_json('simulation/renderoption.json')
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    print(param.extrinsic, param.intrinsic)
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = visualization.Visualizer()
    vis.create_window(window_name='vis', width=1080, height=720)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    print(param.extrinsic, param.intrinsic)
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json('simulation/renderoption.json')
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    object_cls = objects['mug']
    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()

    save_view_point(object_mesh, "simulation/BV_1440.json")  # 保存好得json文件位置
    load_view_point(object_mesh, "simulation/BV_1440.json")  # 加载修改时较后的pcd文件