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
from myutils.assets_config import assets
import open3d as o3d
from myutils.utils import *
import random


device = "cuda"
object_cls = objects['mug']

poses = np.loadtxt('obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_raw.txt')
model = torch.load('prediction/classify/trained_models/' + object_cls.name + '/uncluster_noisy.pkl')

model.eval()

# Coordinate
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# Visualize
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='vis')
vis.add_geometry(coordinate)

object = assets(mesh=object_cls.init_transform())
vis.add_geometry(object.mesh)

hand = assets(mesh=load_mano())
vis.add_geometry(hand.mesh)
target_hand = assets(mesh=load_mano())
target_hand.mesh.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
vis.add_geometry(target_hand.mesh)
pred_hand = assets(mesh=load_mano())
pred_hand.mesh.paint_uniform_color([250 / 255, 127 / 255, 111 / 255])
vis.add_geometry(pred_hand.mesh)


if __name__ == "__main__":    
    # dataset_eval()
    file_path = 'experiment/data/ShiXu/mug_3_log_hand.txt'
    log_hand = np.loadtxt(file_path)
    target_gpose = np.array(log_hand[0])
    TF_oh = log_hand[1:]
    X = torch.from_numpy(TF_oh).type(torch.FloatTensor)
    print(len(X))
    X = X.to(device)
    Pred = []
    with torch.no_grad():
        for x in X:
            pred = model(x)
            Pred.append(pred.argmax(0).item())
    while True:
        for i, idx in enumerate(Pred):
            # In world coordinate
            hand_pose = np.array(TF_oh[i])
            gpose = poses[idx]

            hand.update_transform(hand_pose)
            pred_hand.update_transform(gpose)
            target_hand.update_transform(target_gpose)

            # ========================= visualize in open3d ==================== #
            vis.update_geometry(hand.mesh)
            vis.update_geometry(pred_hand.mesh)
            vis.update_geometry(target_hand.mesh)

            # 更新窗口
            vis.poll_events()
            vis.update_renderer()

    
