"""
Transform world coordinate into object coordinate, and save as txt file.
The transforms include the whole motion (trajectory and grasp pose)
"""
import os
from myutils.utils import *
import numpy as np
from myutils.object_config import objects
import shutil
from myutils.add_gauss_noise import add_pose_noise, add_gaussian_noise

add_noise = False


if __name__ == "__main__":
    object_cls = objects['mug']
    path = 'mocap/' + object_cls.name + '/'
    # Saving path
    save_path = 'obj_coordinate/' + object_cls.name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    # Source files
    source_files = os.listdir(path)
    source_files.sort()
    for i, file in enumerate(source_files):
        file_path = path + file
        Q_wh, T_wh, Q_wo, T_wo, num_frame = read_data(file_path)
        Q_oh, T_oh, TF_oh = sequence_coordinate_transform(Q_wh, T_wh, Q_wo, T_wo, num_frame)
        # If cutting is needed
        length = TF_oh.shape[0]
        cutted_start = int(0.2 * length)
        cutted_end = int(0.7 * length)
        gpose = TF_oh[-1, :].reshape(1, 7)
        sample_interval = 5
        # sample every sample_interval points
        cutted_TF_oh = np.concatenate((TF_oh[cutted_start:cutted_end:sample_interval, :], gpose), axis=0)
        if add_noise:
            noisy_TF_oh = add_pose_noise(cutted_TF_oh, std_q=0.1, std_t=0.01, sample=2)
            np.savetxt(save_path + file[:-3] + 'txt', noisy_TF_oh)
        else:
            np.savetxt(save_path + file[:-3] + 'txt', cutted_TF_oh)

    # Rotate Expansion
    if object_cls.rotate_expansion == 180:
        save_rotate_expansion(save_path, degree=180)
    elif object_cls.rotate_expansion == 90:
        save_rotate_expansion(save_path, degree=90)
        save_rotate_expansion(save_path, degree=180)