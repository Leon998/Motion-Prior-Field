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


device = "cuda"
object_cls = objects['mug']
position_active = False
# if position_active:
#     poses = np.loadtxt('../obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_label_avg_' + str(object_cls.g_clusters) + '.txt')
#     model = torch.load('classify/trained_models/' + object_cls.name + '/position_' + str(object_cls.g_clusters) +'.pkl')
# else:
#     poses = np.loadtxt('../obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_label_avg_' + str(object_cls.g_clusters) + '.txt')
#     model = torch.load('classify/trained_models/' + object_cls.name + '/pose_' + str(object_cls.g_clusters) + '.pkl')

poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/noisy_diverse.pkl')
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
    path = 'mocap/evaluation/' + object_cls.name + '_eval/rand_009.csv'
    # path = '../mocap/mug/handle_009.csv'
    Q_wh, T_wh, Q_wo, T_wo, num_frame = read_data(path)
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
    Real = int(path[-7:-4])
    print(Real)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i, idx in enumerate(Pred):
        if i % 10 == 0:
            pose = TF_oh[i]
            start_pose = hand_transform(pose, init_hand)
            gpose = poses[idx]
            pred_gpose = hand_transform(gpose, init_hand)
            pred_gpose.paint_uniform_color([150/255, 195/255, 125/255])
            vis.clear_geometries()
            vis.add_geometry(start_pose)
            vis.add_geometry(pred_gpose)
            vis.add_geometry(coordinate)
            vis.add_geometry(object_mesh)

            ctr = vis.get_view_control()
            ctr.set_front([0.62896083853874529, 0.48892933979393521, 0.60444715589810261])
            ctr.set_lookat([0.074898500367999096, 0.080563278711616546, 0.07629557461037835])
            ctr.set_up([-0.35855661201677358, 0.87229021616299163, -0.3324859918333018])
            ctr.set_zoom(1.1)

            # 更新窗口
            vis.poll_events()
            vis.update_renderer()

            # 截图
            # vis.capture_screen_image('prediction/classify/img/'+ 'sequence_' + str(i) + '.png')
