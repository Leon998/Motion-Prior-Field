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
from save_gpose_average import save_gpose_avg

add_noise = True


def label_stack(label, num_frame):
    labels = np.zeros((1, 1))
    for i in range(num_frame):
        labels = np.concatenate((labels, label), axis=0)
    labels = labels[1:, :]
    return labels


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
        cutted_start = int(0.1 * length)
        cutted_end = int(0.7 * length)
        gpose = TF_oh[-1, :].reshape(1, 7)
        sample_interval = 1
        # sample every sample_interval points
        cutted_TF_oh = np.concatenate((TF_oh[cutted_start:cutted_end:sample_interval, :], gpose), axis=0)
        if add_noise:
            noisy_TF_oh = add_pose_noise(cutted_TF_oh, std_q=0.15, std_t=0.015, sample=3)
            np.savetxt(save_path + file[:-3] + 'txt', noisy_TF_oh)
        else:
            np.savetxt(save_path + file[:-3] + 'txt', cutted_TF_oh)

    # Rotate Expansion
    if object_cls.rotate_expansion == 180:
        save_rotate_expansion(save_path, degree=180)
    elif object_cls.rotate_expansion == 90:
        save_rotate_expansion(save_path, degree=90)
        save_rotate_expansion(save_path, degree=180)

    # ====================================== Save gposes gtypes ============================== #
    path = 'obj_coordinate/' + object_cls.name + '/'
    q_grasps_oh, t_grasps_oh, tf_grasps_oh, grasp_type_names = grasp_integrate(path, object_cls.grasp_types)
    # Saving pose information
    save_path = 'obj_coordinate/pcd_gposes/' + object_cls.name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.savetxt(save_path + '/' + 'gposes_raw.txt', tf_grasps_oh)
    # Saving grasp type index_list
    with open(save_path + '/' + 'gtypes.txt', 'w') as f:
        for gtype in grasp_type_names:
            f.write(gtype + '\n')
    # ======================================================================================= #
    Traj_TF_oh_labeled = np.zeros((1, 8))
    gposes_labeled = np.zeros((1, 1))
    gposes_path = save_path + '/' + 'gposes_raw.txt'
    gtypes_path = save_path + '/' + 'gtypes.txt'
    field_path = 'obj_coordinate/pcd_field/' + object_cls.name
    if not os.path.exists(field_path):
        os.mkdir(field_path)
    traj_path = 'obj_coordinate/pcd_trajs_labeled/' + object_cls.name
    if not os.path.exists(traj_path):
        os.mkdir(traj_path)
    for j, gtype in enumerate(object_cls.grasp_types):
        # j is the grasp type index_list
        gtype_indices, gtype_poses = gtype_extract(gtype, gposes_path, gtypes_path)
        # print(gtype_indices)
        # ========================== Clustering and saving labels ========================== #
        num_clusters = object_cls.g_clusters
        fig, ax, label = pose_cluster(gtype_poses, num_clusters=num_clusters)
        # fig, ax, label = position_cluster(gtype_poses[:, 4:], num_clusters=object_cls.g_clusters)
        print('label: ', label)
        new_label = []
        # ============= If dynamic clusters is needed, change here ======================== #
        for idx in range(len(label)):
            # tmp = np.concatenate((gtype, glabel), axis=1)
            l = j * num_clusters + label[idx]
            new_label.append(l)
            tmp = np.array(l).reshape(1, 1)
            gposes_labeled = np.concatenate((gposes_labeled, tmp), axis=0)
        # ================================================================================= #
        np.savetxt(save_path + '/' + str(gtype) + '_label.txt', new_label, fmt="%i")
        # Saving trajectories
        path = 'obj_coordinate/' + object_cls.name + '/'
        files = os.listdir(path)
        files.sort()  # Sort all the files in order
        gtype_files = []
        for idx in gtype_indices:
            gtype_files.append(files[idx])
        gtype_pcd = np.zeros((1, 6))  # Storing locations of each point in one gtype
        for i, file in enumerate(gtype_files):
            file_path = path + file
            hand_poses = np.loadtxt(file_path)
            Q_oh = hand_poses[:, :4]
            T_oh = hand_poses[:, 4:]
            TF_oh = hand_poses[:, :]
            num_frame = hand_poses.shape[0]
            # ============= If dynamic clusters is needed, change here ======================== #
            single_label = j * num_clusters + label[i]
            labels = label_stack(np.array(single_label).reshape(1, 1), num_frame)
            # ================================================================================= #
            # Store colored gtype_pcd
            for point in T_oh:
                color = colorlib[int(single_label)].reshape(1, 3)
                tmp = np.concatenate((point.reshape(1, 3), color), axis=1)
                gtype_pcd = np.concatenate((gtype_pcd, tmp), axis=0)
            #
            tmp = np.concatenate((TF_oh, labels), axis=1)
            TF_oh = np.concatenate((TF_oh, labels), axis=1)
            np.savetxt(traj_path + '/' + file[:-3] + 'txt', TF_oh)
            Traj_TF_oh_labeled = np.concatenate((Traj_TF_oh_labeled, tmp), axis=0)
        # Save sub-field
        gtype_pcd = gtype_pcd[1:, :]
        np.savetxt(field_path + '/' + gtype + '.xyzrgb', gtype_pcd)

    # End of all trajectory points, all of them are stacked together
    gposes_labeled = gposes_labeled[1:, :]
    # ======================================= Save gpose_label ======================================== #
    np.savetxt(save_path + '/' + 'gposes_label.txt', gposes_labeled, fmt="%i")
    # ====================================== Save gpose_average ======================================= #
    save_gpose_avg(object_cls)
    # ======================================= Save TF_field ======================================== #
    Traj_TF_oh_labeled = Traj_TF_oh_labeled[1:, :]
    np.savetxt(field_path + '/' + 'TF_field.txt', Traj_TF_oh_labeled)
    np.savetxt('prediction/classify/training_data/' + object_cls.name + '_field.txt', Traj_TF_oh_labeled)
    # ====================================== Save T_colored ====================================== #
    Traj_T_oh_colored = np.zeros((1, 6))
    for point in Traj_TF_oh_labeled:
        position = point[4:7]
        label = point[7]
        color_idx = int(label)
        color = colorlib[color_idx]
        t_oh_colored = np.concatenate((position.reshape((1, 3)), color.reshape((1, 3))), axis=1)
        Traj_T_oh_colored = np.concatenate((Traj_T_oh_colored, t_oh_colored), axis=0)
    Traj_T_oh_colored = Traj_T_oh_colored[1:, :]
    np.savetxt(field_path + '/' + 'T_colored.xyzrgb', Traj_T_oh_colored)
    # ====================================== Save sub-field ===================================== #
    # pcd = Traj_TF_oh_labeled[:, 4:7]
    # label = Traj_TF_oh_labeled[:, 7]
    # gtypes = object_cls.grasp_types
    # for gtype in gtypes:
    #     save_sub_field(pcd, label, gtype, field_path)