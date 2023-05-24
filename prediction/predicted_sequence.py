import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from myutils.object_config import objects
from myutils.hand_config import *
import open3d as o3d
from myutils.utils import *
import random


device = "cuda"
object_cls = objects['tomato_soup_can']
# cluster
# poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_label_avg_' + str(object_cls.g_clusters) + '.txt')
# model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/pose_' + str(object_cls.g_clusters) + '.pkl')

# uncluster
poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')

model.eval()

# Coordinate
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
# Object
object_mesh = object_cls.init_transform()
# Hand
init_hand = load_mano()
meshes = [coordinate, object_mesh]


if __name__ == "__main__":
    # dataset_eval()
    file_path = 'mocap/' + object_cls.name + '/'
    files = os.listdir(file_path)
    file = random.choice(files)
    Q_wh, T_wh, Q_wo, T_wo, num_frame = read_data(file_path + file)
    Q_oh, T_oh, TF_oh = sequence_coordinate_transform(Q_wh, T_wh, Q_wo, T_wo, num_frame)
    length = TF_oh.shape[0]
    X = torch.from_numpy(TF_oh).type(torch.FloatTensor)
    print(len(X))
    X = X.to(device)
    Pred = []
    with torch.no_grad():
        for x in X:
            pred = model(x)
            Pred.append(pred.argmax(0).item())
    print(Pred)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis', width=1440, height=1080)
    for i, idx in enumerate(Pred):
        if i % 10 == 0:
            pose = TF_oh[i]
            start_pose = hand_transform(pose, init_hand)
            gpose = poses[idx]
            pred_gpose = hand_transform(gpose, init_hand)
            pred_gpose.paint_uniform_color([150/255, 195/255, 125/255])

            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('simulation/BV_1440.json')
            vis.clear_geometries()
            vis.add_geometry(start_pose)
            vis.add_geometry(pred_gpose)
            vis.add_geometry(coordinate)
            vis.add_geometry(object_mesh)
            vis.get_render_option().load_from_json('simulation/renderoption.json')
            ctr.convert_from_pinhole_camera_parameters(param)

            # 更新窗口
            vis.poll_events()
            vis.update_renderer()

            # 截图
            # vis.capture_screen_image('prediction/classify/img/'+ 'sequence_' + str(i) + '.png')
