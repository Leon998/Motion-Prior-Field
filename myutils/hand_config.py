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

# def incremental_hand_transform(hand, hand_pose, translation, rotation):
#     last_translation = np.array(translation)
#     translation = np.array(hand_pose[4:])
#     delta_translation = translation - last_translation
    
#     last_rotation = rotation
#     rotation = tuple((hand_pose[3], hand_pose[0], hand_pose[1], hand_pose[2]))
#     last_R = hand.get_rotation_matrix_from_quaternion(last_rotation)
#     current_R = hand.get_rotation_matrix_from_quaternion(rotation)
#     delta_R = (current_R).dot(np.linalg.inv(last_R))
#     return translation, rotation, delta_translation, delta_R

if __name__ == "__main__":
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    init_hand = load_mano()
    hand_bias = init_hand.get_center()
    # print(hand_bias)

    # ====== basic rotation and translation test ====== #
    # new_hand = copy.deepcopy(init_hand)
    # translation = (0.2, 0, 0)
    # print(translation)
    # new_hand.translate(translation, relative=True)
    # rotation = (0, np.pi / 2, np.pi / 2)
    # R1 = new_hand.get_rotation_matrix_from_zyx(rotation)
    # new_hand.rotate(R1, center=translation)
    # print(new_hand.get_center())

    o3d.visualization.draw_geometries([coordinate, init_hand])
    

    # ========== wrist joint calculate test ============ #
    mid_hand = copy.deepcopy(init_hand)
    euler1 = [30, 20, 60]
    r1 = R.from_euler('zyx', euler1, degrees=True).as_matrix()
    mid_hand.rotate(r1, center=(0, 0, 0))

    target_hand = copy.deepcopy(init_hand)
    euler2 = [60, 20, 60]
    r2 = R.from_euler('zyx', euler2, degrees=True).as_matrix()
    target_hand.rotate(r2, center=(0, 0, 0))

    # 理论上我需要的关节旋转角为[30, 0, 0]，但是变换的结果肯定不对
    # euler_transform = [30, 0, 0]
    # r_wrist = R.from_euler('zyx', euler_transform, degrees=True).as_matrix()
    # new_hand.rotate(r_wrist, center=(0, 0, 0))

    # 这样变换结果肯定对，但是关节旋转角又不是[30, 0, 0]了
    # r_wrist = (r2).dot(np.linalg.inv(r1))
    # euler_transform = R.from_matrix(r_wrist).as_euler('zyx', degrees=True)
    # print(euler_transform)  
    # new_hand.rotate(r_wrist, center=(0, 0, 0))

    # 因此，先将目标手用当前手的pose变换到init状态
    norm_target_hand = copy.deepcopy(target_hand)
    norm_target_pose = np.linalg.inv(r1)
    norm_target_hand.rotate(norm_target_pose, center=(0, 0, 0))
    # 此时再算变换角，这里指的是从init_hand到此时的hand
    r_wrist = np.linalg.inv(r1).dot(r2)  # 莫名其妙，r2乘r1的逆就不对，反过来就对了。猜测是因为旋转矩阵都是往左边乘的
    euler_joint = R.from_matrix(r_wrist).as_euler('zyx', degrees=True)
    print(euler_joint)
    new_hand = copy.deepcopy(init_hand)
    new_hand.rotate(r_wrist, center=(0, 0, 0))


    o3d.visualization.draw_geometries([coordinate, mid_hand, target_hand, norm_target_hand, new_hand])

    

