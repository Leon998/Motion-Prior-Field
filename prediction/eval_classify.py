import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_config import *
from myutils.object_config import objects
from myutils.hand_config import *
import open3d as o3d
from myutils.utils import *


device = "cuda"
object_cls = objects['mug']
model = torch.load('classify/trained_models/' + object_cls.name + '_' + str(object_cls.g_clusters) +'.pkl')
model.eval()

# Coordinate
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
# Object
object_mesh = object_cls.init_transform()
# Hand
init_hand = load_mano()
meshes = [coordinate, object_mesh]

poses_avg = np.loadtxt('../obj_coordinate/pcd_gposes/' + object_cls.name + '/gposes_label_avg.txt')


if __name__ == "__main__":
    path = 'classify/training_data/' + object_cls.name + '_field.txt'
    batch_size = 64
    # Get cpu or gpu device for training.
    train_set, validate_set, _, _ = data_loading(path, batch_size)
    num_example = 1
    X, Y = validate_set[:num_example][0], validate_set[:num_example][1]
    X, Y = X.to(device), Y.to(device)
    Pred = []
    with torch.no_grad():
        for x in X:
            pred = model(x)
            # Pred.append(pred.argmax(0).item())
            indices = torch.topk(pred, 3).indices
            Pred = indices.cpu().numpy().tolist()

    Start_pose = X.cpu().numpy()
    Real = Y.cpu().numpy().tolist()
    print(Real, '\n', Pred)

    # Visualize
    for pose in Start_pose:
        start_gpose = hand_transform(pose, init_hand)
        meshes.append(start_gpose)
    # for idx in Real:
    #     pose = poses[idx][:-1]
    #     real_gpose = hand_transform(pose, init_hand)
    #     real_gpose.paint_uniform_color([0 / 255, 255 / 255, 0 / 255])
    #     meshes.append(real_gpose)
    for i, idx in enumerate(Pred):
        pose = poses_avg[idx]
        pred_gpose = hand_transform(pose, init_hand)
        if idx == Real[0]:
            print("ass")
            pred_gpose.paint_uniform_color([0 / 255, 255 / 255, 0 / 255])
            meshes.append(pred_gpose)
        else:
            pred_gpose.paint_uniform_color([0.4 * i, 0.4 * i, 0.4 * i])
            meshes.append(pred_gpose)

    o3d.visualization.draw_geometries(meshes)
