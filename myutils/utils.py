import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from myutils.object_config import *

"""
Q, T mean one trial's data stack up by time (frames)
q, t mean an instant data in one trial
q_grasps, t_grasps mean all grasp poses stack up by trials
"""


def read_data(file_name):
    """
    Extract rotation (quaternion type) and translation, and transform into numpy array
    Parameters
    ----------
    file_name : name of csv file
    Returns
    ----------
    Q_wh, T_wh, Q_wo, T_wo, num_frame : quaternion and translation, stack vertically by time
        In shape of (num_frame, 4), (num_frame, 3), (num_frame, 4), (num_frame, 3)
    """
    df_raw = pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], skiprows=6)
    data_raw = np.array(df_raw)
    num_frame = data_raw.shape[0]
    Q_wh = data_raw[:, :4]  # X,Y,Z,W
    T_wh = data_raw[:, 4:7]  # X,Y,Z
    Q_wo = data_raw[:, 7:11]  # X,Y,Z,W
    T_wo = data_raw[:, 11:14]  # X,Y,Z
    return Q_wh, T_wh, Q_wo, T_wo, num_frame


def extract_grasp(Q_wh, T_wh, Q_wo, T_wo):
    """
    Extract the last frame, which is the grasp pose. In form of object coordinate
    Parameters
    ----------
    Q_wh, T_wh, Q_wo, T_wo : Sequence of hand and object pose from the csv
    Returns
    ----------
    q_oh, t_oh, tf_oh : quaternion and translation of hand in object coordinate, tf_oh concatenate q_oh and t_oh
    """
    ## Last frame
    # hand
    q_wh = Q_wh[-1, :]
    t_wh = T_wh[-1, :]
    # object
    q_wo = Q_wo[-1, :]
    t_wo = T_wo[-1, :]

    ## Transformation
    q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
    # print(q_wo, '\n', t_wo, '\n', q_wh, '\n', t_wh)
    # print(('==========================='))
    # print(q_oh, '\n', t_oh, '\n', tf_oh)
    return q_oh, t_oh, tf_oh


def coordinate_transform(q_wh, t_wh, q_wo, t_wo):
    """
    Transform world coordinate to object coordinate
    Parameters
    ----------
    q_wo, t_wo, q_wh, t_wh : quaternion and translation of object and hand respectively in world coordinate
    Returns
    ----------
    q_oh, t_oh, tf_oh : quaternion and translation of hand in object coordinate, tf_oh concatenate q_oh and t_oh
    """
    r_wo = R.from_quat(q_wo).as_matrix()  # Get object rotation matrix from quaternion
    r_wh = R.from_quat(q_wh).as_matrix()  # Get hand rotation matrix from quaternion
    # Transform
    r_oh = (np.linalg.inv(r_wo)).dot(r_wh)
    # r_oh = (r_wh).dot(np.linalg.inv(r_wo))
    q_oh = R.from_matrix(r_oh).as_quat()
    # t_oh = t_wh + (-t_wo)  # This is right only when object is fixed, and human hand pivots
    t_oh = np.linalg.inv(r_wo).dot(
        t_wh + (-t_wo))  # Guess, pivot the object is right? Note that r_wo should be inversed
    tf_oh = np.concatenate((q_oh, t_oh), axis=0)
    return q_oh, t_oh, tf_oh


def sequence_coordinate_transform(Q_wh, T_wh, Q_wo, T_wo, num_frame):
    """
    Coordinate tramsform of each frame in a sequence
    Parameters
    ----------
    Q_wh, T_wh, Q_wo, T_wo : Sequence of hand and object pose from the csv
    Returns
    ----------
    Q_oh, T_oh, TF_oh : Transformed hand pose w.r.t. object
    """
    Q_oh = np.zeros((1, 4))
    T_oh = np.zeros((1, 3))
    TF_oh = np.zeros((1, 7))
    for i in range(num_frame):
        # hand
        q_wh = Q_wh[i, :]
        t_wh = T_wh[i, :]
        # object
        q_wo = Q_wo[i, :]
        t_wo = T_wo[i, :]
        # Transformation
        q_oh, t_oh, tf_oh = coordinate_transform(q_wh, t_wh, q_wo, t_wo)
        # Concatenate
        Q_oh = np.concatenate((Q_oh, q_oh.reshape(1, 4)), axis=0)
        T_oh = np.concatenate((T_oh, t_oh.reshape(1, 3)), axis=0)
        TF_oh = np.concatenate((TF_oh, tf_oh.reshape(1, 7)), axis=0)
    Q_oh = Q_oh[1:, :]
    T_oh = T_oh[1:, :]
    TF_oh = TF_oh[1:, :]
    return Q_oh, T_oh, TF_oh


def rotate_expansion(q_oh, t_oh, tf_oh, degree=180):
    r = R.from_euler('y', degree, degrees=True).as_matrix()
    r_oh = R.from_quat(q_oh).as_matrix()
    new_r_oh = r.dot(r_oh)
    new_q_oh = R.from_matrix(new_r_oh).as_quat()
    # new_t_oh = -t_oh
    # new_t_oh[1] = -new_t_oh[1]
    new_t_oh = r.dot(t_oh)
    new_tf_oh = np.concatenate((new_q_oh, new_t_oh), axis=0)
    return new_q_oh, new_t_oh, new_tf_oh


def sequence_rotate_expansion(Q_oh, T_oh, TF_oh, num_frame, degree=180):
    new_Q_oh = np.zeros((1, 4))
    new_T_oh = np.zeros((1, 3))
    new_TF_oh = np.zeros((1, 7))
    for i in range(num_frame):
        q_oh = Q_oh[i, :]
        t_oh = T_oh[i, :]
        tf_oh = TF_oh[i, :]
        new_q_oh, new_t_oh, new_tf_oh = rotate_expansion(q_oh, t_oh, tf_oh, degree)
        # Concatenate
        new_Q_oh = np.concatenate((new_Q_oh, new_q_oh.reshape(1, 4)), axis=0)
        new_T_oh = np.concatenate((new_T_oh, new_t_oh.reshape(1, 3)), axis=0)
        new_TF_oh = np.concatenate((new_TF_oh, new_tf_oh.reshape(1, 7)), axis=0)
    new_Q_oh = new_Q_oh[1:, :]
    new_T_oh = new_T_oh[1:, :]
    new_TF_oh = new_TF_oh[1:, :]
    return new_Q_oh, new_T_oh, new_TF_oh


def save_rotate_expansion(save_path, degree=180):
    files = os.listdir(save_path)
    files.sort()
    gtype_log = {}
    for file in files:
        gtype = file[:-8]
        gtype_log[gtype] = file[-7:-4]
    print(gtype_log)
    for i, file in enumerate(files):
        gtype = file[:-8]
        index = int(file[-7:-4])  # current index_list
        file_path = save_path + file
        origin_pose = np.loadtxt(file_path)
        # # ================================ Trajectory test =============================== #
        Q_oh = origin_pose[:, :4]
        T_oh = origin_pose[:, 4:]
        TF_oh = origin_pose[:, :]
        num_frame = origin_pose.shape[0]
        new_Q_oh, new_T_oh, new_TF_oh = sequence_rotate_expansion(Q_oh, T_oh, TF_oh, num_frame, degree)
        new_index = int(gtype_log[gtype]) + index + 1
        if new_index <= 9:
            zeros = '00'
        elif new_index <= 99:
            zeros = '0'
        else:
            zeros = ''
        print(str(gtype) + '_' + zeros + str(new_index) + '.txt')
        np.savetxt(save_path + str(gtype) + '_' + zeros + str(new_index) + '.txt', new_TF_oh)

def grasp_integrate_archive(path, grasp_types):
    """
    Stack all the grasp pose Transformation, Quaternion, Translation of each trial
    Parameters
    ----------
    path : file path of all tracking files
    Returns
    ----------
    q_grasps_oh, t_grasps_oh, tf_grasps_oh : All the grasp poses Quaternion, Translation, and Transformation of each trial
    """
    files = os.listdir(path)
    files.sort()  # Sort all the files in order
    # print(files)
    q_grasps_oh = np.zeros((1, 4))
    t_grasps_oh = np.zeros((1, 3))
    tf_grasps_oh = np.zeros((1, 7))
    grasp_type_names = []
    for file in files:
        file_path = path + file
        Q_wh, T_wh, Q_wo, T_wo, _ = read_data(file_path)
        q_oh, t_oh, tf_oh = extract_grasp(Q_wh, T_wh, Q_wo, T_wo)

        q_grasps_oh = np.concatenate((q_grasps_oh, q_oh.reshape(1, 4)), axis=0)
        t_grasps_oh = np.concatenate((t_grasps_oh, t_oh.reshape(1, 3)), axis=0)
        tf_grasps_oh = np.concatenate((tf_grasps_oh, tf_oh.reshape(1, 7)), axis=0)

        for grasp_type in grasp_types:
            if file[:-8] == grasp_type:
                grasp_type_names.append(grasp_type)

    q_grasps_oh = q_grasps_oh[1:, :]  # q_grasps_oh contains quaternion
    t_grasps_oh = t_grasps_oh[1:, :]  # tf_grasps_oh contains translate
    tf_grasps_oh = tf_grasps_oh[1:, :]  # tf_grasps_oh contains all the hand transformations
    # w.r.t the object, including quaternion and translate
    return q_grasps_oh, t_grasps_oh, tf_grasps_oh, grasp_type_names


def grasp_integrate(path, grasp_types):
    """
        Stack all the grasp pose Transformation, Quaternion, Translation of each trial
        Parameters
        ----------
        path : file path of all tracking files (object coordinate)
        Returns
        ----------
        q_grasps_oh, t_grasps_oh, tf_grasps_oh : All the grasp poses Quaternion, Translation, and Transformation of each trial
        """
    files = os.listdir(path)
    files.sort()  # Sort all the files in order
    # print(files)
    q_grasps_oh = np.zeros((1, 4))
    t_grasps_oh = np.zeros((1, 3))
    tf_grasps_oh = np.zeros((1, 7))
    grasp_type_names = []
    for file in files:
        file_path = path + file
        hand_poses = np.loadtxt(file_path)
        q_oh = hand_poses[-1, :4]
        t_oh = hand_poses[-1, 4:]
        tf_oh = hand_poses[-1, :]

        q_grasps_oh = np.concatenate((q_grasps_oh, q_oh.reshape(1, 4)), axis=0)
        t_grasps_oh = np.concatenate((t_grasps_oh, t_oh.reshape(1, 3)), axis=0)
        tf_grasps_oh = np.concatenate((tf_grasps_oh, tf_oh.reshape(1, 7)), axis=0)

        for grasp_type in grasp_types:
            if file[:-8] == grasp_type:
                grasp_type_names.append(grasp_type)

    q_grasps_oh = q_grasps_oh[1:, :]  # q_grasps_oh contains quaternion
    t_grasps_oh = t_grasps_oh[1:, :]  # tf_grasps_oh contains translate
    tf_grasps_oh = tf_grasps_oh[1:, :]  # tf_grasps_oh contains all the hand transformations
    # w.r.t the object, including quaternion and translate
    return q_grasps_oh, t_grasps_oh, tf_grasps_oh, grasp_type_names


def grasp_type_index(human_label_path):
    """
    Extract grasp types and return a dictionary, containing each type and file index_list respectively
    Parameters
    ----------
    human_label_path

    Returns
    -------
    grasp_type_lib: a dictionary containing each type and file index_list respectively
    """
    grasp_type_names = []
    grasp_type_lib = {}
    f = open(human_label_path, "r")
    lines = f.readlines()
    i = 0
    for line in lines:
        line = line.strip('\n')  # 删除\n
        grasp_type_names.append(line)
        if grasp_type_names.count(line) == 1:
            grasp_type_lib[line] = [i]
        else:
            grasp_type_lib[line].append(i)
        i += 1
    return grasp_type_lib


def gtype_extract(gtype, gposes_path, gtypes_path):
    """

    Parameters
    ----------
    gtype: current grasp type
    gposes_path: path to file containing all the raw grasp poses
    gtypes_path: path to file of grasp type names

    Returns
    -------
    gtype_indices: grasp type indices of current grasp type
    gtype_poses: grasp pose of current grasp type
    """
    # Reading pose
    grasp_poses = np.loadtxt(gposes_path)
    # Reading grasp type name
    grasp_type_lib = grasp_type_index(gtypes_path)
    # Extract grasp types
    gtype_indices = grasp_type_lib[gtype]
    gtype_poses = grasp_poses[gtype_indices]
    return gtype_indices, gtype_poses


def position_cluster(t_grasps_oh, num_clusters=5):
    from sklearn.cluster import AgglomerativeClustering

    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward").fit(t_grasps_oh)
    label = ward.labels_
    print(f"Number of points: {label.size}")

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0, 0, 0.95, 1])
    for l in np.unique(label):
        ax.scatter(
            t_grasps_oh[label == l, 0],
            t_grasps_oh[label == l, 1],
            t_grasps_oh[label == l, 2],
            color=plt.cm.jet(float(l) / np.max(label + 1)),
            s=20,
            edgecolor="k",
        )
    # plt.show()
    return fig, ax, label


def pose_cluster(tf_grasps_oh, num_clusters=4):
    from sklearn.cluster import AgglomerativeClustering

    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward").fit(tf_grasps_oh)
    label = ward.labels_
    print(f"Number of points: {label.size}")

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0, 0, 0.95, 1])
    for l in np.unique(label):
        ax.scatter(
            tf_grasps_oh[label == l, 4],
            tf_grasps_oh[label == l, 5],
            tf_grasps_oh[label == l, 6],
            color=plt.cm.jet(float(l) / np.max(label + 1)),
            s=20,
            edgecolor="k",
        )
    # plt.show()
    return fig, ax, label
