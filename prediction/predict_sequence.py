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
subject = 'subjects/'
object_cls = objects['mug']

poses = np.loadtxt('prediction/module/' + subject +'gposes/' + object_cls.name + '/gposes_raw.txt')
model = torch.load('prediction/module/' + subject +'trained_models/' + object_cls.name + '/' + subject[:-1] +'_uncluster_noisy.pkl')

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
pred_hand = assets(mesh=load_mano())
pred_hand.mesh.paint_uniform_color([150 / 255, 195 / 255, 125 / 255])
vis.add_geometry(hand.mesh)
vis.add_geometry(pred_hand.mesh)


if __name__ == "__main__":    
    # dataset_eval()
    file_path = 'mocap/' + subject + object_cls.name + '/'
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
    
    for i, idx in enumerate(Pred):
        # In world coordinate
        hand_pose = np.concatenate((Q_wh[i], T_wh[i]))
        gpose = gpose2wdc(poses[idx], Q_wo[i], T_wo[i]) 
        object_pose = np.concatenate((Q_wo[i], T_wo[i]))

        hand.update_transform(hand_pose)
        pred_hand.update_transform(gpose)
        object.update_transform(object_pose)
        
        # ========================= visualize in open3d ==================== #
        vis.update_geometry(hand.mesh)
        vis.update_geometry(pred_hand.mesh)
        vis.update_geometry(object.mesh)

        # 更新窗口
        vis.poll_events()
        vis.update_renderer()

    
