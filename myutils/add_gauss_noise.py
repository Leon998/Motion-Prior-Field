import os
import sys
sys.path.append(os.getcwd())
from myutils.utils import *
import numpy as np
from myutils.object_config import objects, colorlib
from myutils.hand_config import *
import open3d as o3d


def add_gaussian_noise(input, mean=0, std=0.2):
    noise = np.random.normal(mean, std, input.shape)
    noisy_input = input + noise
    return noisy_input

def noise_hand(hand_pose, std_q=0.05, std_t=0.015):
    noisy_q = add_gaussian_noise(hand_pose[:4], 0, std_q)
    noisy_t = add_gaussian_noise(hand_pose[4:], 0, std_t)
    noisy_hand_pose = np.concatenate((noisy_q, noisy_t))
    return noisy_hand_pose


def add_pose_noise(cutted_TF_oh, std_q=0.08, std_t=0.008, sample=3):
    tmp = np.zeros((1, 7))
    for hand_pose in cutted_TF_oh[:-1]:
        for _ in range(sample):
            noisy_q = add_gaussian_noise(hand_pose[:4], 0, std_q)
            noisy_t = add_gaussian_noise(hand_pose[4:], 0, std_t)
            noisy_hand_pose = np.concatenate((noisy_q, noisy_t))
            tmp = np.concatenate((tmp, noisy_hand_pose.reshape(1, 7)), axis=0)
    tmp = tmp[1:]
    noisy_TF_oh = np.concatenate((tmp, cutted_TF_oh), axis=0)
    return noisy_TF_oh


def single_hand_add_noise(pose, init_hand, color, meshes, std_q=0.1, std_t=0.01, sample=1):
    for _ in range(sample):
        noisy_q = add_gaussian_noise(pose[:4], 0, std_q)
        noisy_t = add_gaussian_noise(pose[4:], 0, std_t)
        noisy_hand_pose = np.concatenate((noisy_q, noisy_t))
        hand_noise = hand_transform(noisy_hand_pose, init_hand)
        if color is not None:
            hand_noise.paint_uniform_color(color)
        meshes.append(hand_noise)


if __name__ == "__main__":
    object_cls = objects['mug']
    path = 'mocap/' + object_cls.name + '/'
    # Source files
    source_files = os.listdir(path)
    source_files.sort()
    files = [source_files[2], source_files[3], source_files[4]]

    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()
    meshes = [coordinate, object_mesh]
    for j, file in enumerate(files):
        file_path = path + file
        Q_wh, T_wh, Q_wo, T_wo, num_frame = read_data(file_path)
        Q_oh, T_oh, TF_oh = sequence_coordinate_transform(Q_wh, T_wh, Q_wo, T_wo, num_frame)
        # If cutting is needed
        length = TF_oh.shape[0]
        cutted_start = int(0.2 * length)
        cutted_end = int(0.7 * length)
        gpose = TF_oh[-1, :].reshape(1, 7)
        cutted_TF_oh = np.concatenate((TF_oh[cutted_start:cutted_end, :], gpose), axis=0)

        for i, pose in enumerate(cutted_TF_oh):
            color = colorlib[j]
            std_q, std_t, sample = 0.15, 0.02, 30
            if i % 20 == 0 and i > 0:
                hand_raw = hand_transform(pose, init_hand)
                hand_raw.paint_uniform_color(color)
                meshes.append(hand_raw)
                single_hand_add_noise(pose, init_hand, color, meshes, std_q=std_q, std_t=std_t, sample=sample)
            elif i == len(cutted_TF_oh)-1:
                hand_grasp = hand_transform(pose, init_hand)
                hand_grasp.paint_uniform_color(color)
                meshes.append(hand_grasp)

    o3d.visualization.draw_geometries(meshes,
                                      front=[0.10809381818383093, 0.88692643300126039,-0.44908487940934039],
                                      lookat=[0.2234052048544109, 0.028455189158720258,-0.1370236742460966],
                                      up=[-0.52518785564608772, 0.43449672530496453,0.73170370504810645],
                                      zoom=0.73999999999999999)