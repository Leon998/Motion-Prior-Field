import open3d as o3d
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_mano():
    # hand model
    init_hand_mesh = o3d.io.read_triangle_mesh('models/mano_left.obj')
    init_hand_mesh.compute_vertex_normals()
    init_hand_mesh.translate((0, 0, 0), relative=False)

    # Initial Rotate
    y_rotation = (-np.pi / 4.5, 0, 0)
    Ry = init_hand_mesh.get_rotation_matrix_from_yzx(y_rotation)
    init_hand_mesh.rotate(Ry)
    z_rotation = (0, -np.pi / 4, 0)
    Rz = init_hand_mesh.get_rotation_matrix_from_yzx(z_rotation)
    init_hand_mesh.rotate(Rz)
    x_rotation = (0, 0, -np.pi / 18)
    Rx = init_hand_mesh.get_rotation_matrix_from_yzx(x_rotation)
    init_hand_mesh.rotate(Rx)
    # Initial Translate
    init_hand_translation = (0.09, -0.01, 0)
    init_hand_mesh.translate(init_hand_translation, relative=False)
    return init_hand_mesh


def hand_transform(pose, init_hand):
    """
    Transform the initial hand to a new hand according to the hand pose
    :param pose: quaternion and translation
    :return: hand: a new hand mesh
    """
    hand = copy.deepcopy(init_hand)
    translation = tuple(pose[4:])
    hand.translate(translation, relative=True)
    rotation = tuple((pose[3], pose[0], pose[1], pose[2]))
    R = hand.get_rotation_matrix_from_quaternion(rotation)
    hand.rotate(R, center=translation)
    return hand

def incremental_hand_transform(hand, hand_pose, translation, rotation):
    last_translation = np.array(translation)
    translation = np.array(hand_pose[4:])
    delta_translation = translation - last_translation
    
    last_rotation = rotation
    rotation = tuple((hand_pose[3], hand_pose[0], hand_pose[1], hand_pose[2]))
    last_R = hand.get_rotation_matrix_from_quaternion(last_rotation)
    current_R = hand.get_rotation_matrix_from_quaternion(rotation)
    delta_R = (current_R).dot(np.linalg.inv(last_R))
    return translation, rotation, delta_translation, delta_R

if __name__ == "__main__":
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    init_hand_mesh = load_mano()
    hand_bias = init_hand_mesh.get_center()
    print(hand_bias)

    new_hand = copy.deepcopy(init_hand_mesh)
    translation = (0.2, 0, 0)
    print(translation)
    new_hand.translate(translation, relative=True)

    rotation = (0, np.pi / 2, np.pi / 2)
    R1 = new_hand.get_rotation_matrix_from_yzx(rotation)
    new_hand.rotate(R1, center=translation)
    print(new_hand.get_center())

    o3d.visualization.draw_geometries([coordinate, init_hand_mesh, new_hand])

