from myutils.hand_config import *
from myutils.object_config import colorlib, objects
from myutils.utils import *
import open3d as o3d


if __name__ == "__main__":
    object_cls = objects['cracker_box']
    gtype = 'side'
    save_path = 'obj_coordinate/pcd_gposes/' + object_cls.name
    gposes_path = save_path + '/' + 'gposes_raw.txt'
    gtypes_path = save_path + '/' + 'gtypes.txt'
    gtype_indices, gtype_poses = gtype_extract(gtype, gposes_path, gtypes_path)

    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # Hand
    init_hand = load_mano()

    meshes = [coordinate, object_mesh]
    label = np.loadtxt(save_path + '/' + str(gtype)+'_label.txt')
    for i in range(gtype_poses.shape[0]):
        pose = gtype_poses[i]
        hand_gpose = hand_transform(pose, init_hand)
        hand_gpose.paint_uniform_color(colorlib[int(label[i])])
        meshes.append(hand_gpose)

    field_path = 'obj_coordinate/pcd_field/' + object_cls.name
    pcd = o3d.io.read_point_cloud(field_path + '/' + str(gtype) +'.xyzrgb')
    # pcd = o3d.io.read_point_cloud(field_path + '/' + 'field_position.xyz')
    meshes.append(pcd)
    o3d.visualization.draw_geometries(meshes,
                                      front=[ 0.66825386882419247, 0.59555493871838916, 0.44581507575410106 ],
                                      lookat=[ 0.19713900239960416, 0.20577667615767839, 0.064039819038006374 ],
                                      up=[ -0.49497043196066032, 0.8033143113971537, -0.33119539336952458 ],
                                      zoom=0.54)
