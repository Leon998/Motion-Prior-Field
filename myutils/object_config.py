import open3d as o3d
import numpy as np


class Object:
    def __init__(self,
                 name: object,
                 file_path: object,
                 init_pose: object,
                 grasp_types: object,
                 g_clusters=1,
                 rotate_expansion: object = 0):
        self.name = name
        self.file_path = file_path
        self.init_pose = init_pose
        self.grasp_types = grasp_types
        self.g_clusters = g_clusters
        self.rotate_expansion = rotate_expansion
        # self.color_label = color_label

    def init_transform(self):
        object_mesh = o3d.io.read_triangle_mesh(self.file_path, True)
        R = object_mesh.get_rotation_matrix_from_xyz(self.init_pose)
        object_mesh.rotate(R)
        return object_mesh


colors = [[142, 207, 201],
          [255, 190, 122],
          [250, 127, 111],
          [130, 176, 210],
          [190, 184, 220],
          [153, 153, 153],
          [231, 218, 210],
          [40, 120, 181],
          [154, 201, 219],
          [195, 36, 35],
          [20, 81, 124],
          [231, 239, 245],
          [150, 195, 125],
          [243, 210, 102],
          [196, 151, 178],
          [73, 108, 136],
          [169, 184, 198],
          [169, 144, 126],
          [243, 222, 186],
          [171, 196, 170],
          [103, 93, 80],
          [252, 115, 0],
          [191, 219, 56],
          [31, 138, 112]]

colorlib = []
for color in colors:
    color = np.array(color) / 255
    colorlib.append(color)
for i in range(500):
    colorlib.append(np.array([50, 50, 50]))

objects = {}
PATH = 'models/'

master_chef_can = Object(name='master_chef_can',
                         file_path=PATH + 'ycb_models/002_master_chef_can/textured.obj',
                         init_pose=(-np.pi / 2, 0, -np.pi / 3),
                         grasp_types=['side', 'top'],
                         rotate_expansion=90)
objects['master_chef_can'] = master_chef_can

cracker_box = Object(name='cracker_box',
                     file_path=PATH + 'ycb_models/003_cracker_box/textured.obj',
                     init_pose=(-np.pi / 2, 0, 0),
                     grasp_types=['side', 'top'],
                     rotate_expansion=180,
                     g_clusters=1)
objects['cracker_box'] = cracker_box

sugar_box = Object(name='sugar_box',
                   file_path=PATH + 'ycb_models/004_sugar_box/textured.obj',
                   init_pose=(-np.pi / 2, 0, 0),
                   grasp_types=['side', 'top', 'wide'],
                   rotate_expansion=180)
objects['sugar_box'] = sugar_box

tomato_soup_can = Object(name='tomato_soup_can',
                         file_path=PATH + 'ycb_models/005_tomato_soup_can/textured_simple.obj',
                         init_pose=(-np.pi / 2, 0, -np.pi / 2),
                         grasp_types=['side', 'top'],
                         g_clusters=1,
                         rotate_expansion=90)
objects['tomato_soup_can'] = tomato_soup_can

mustard_bottle = Object(name='mustard_bottle',
                        file_path=PATH + 'ycb_models/006_mustard_bottle/textured_simple.obj',
                        init_pose=(-np.pi / 2, 0, -np.pi / 3),
                        grasp_types=['side1', 'side2', 'top'],
                        g_clusters=1,
                        rotate_expansion=180)
objects['mustard_bottle'] = mustard_bottle

tuna_fish_can = Object(name='tuna_fish_can',
                       file_path=PATH + 'ycb_models/007_tuna_fish_can/textured.obj',
                       init_pose=(-np.pi / 2, 0, -np.pi * 4 / 9),
                       grasp_types=['side', 'top'],
                       rotate_expansion=90)
objects['tuna_fish_can'] = tuna_fish_can

pudding_box = Object(name='pudding_box',
                     file_path=PATH + 'ycb_models/008_pudding_box/textured.obj',
                     init_pose=(0, np.pi / 2, -np.pi * 1.05 / 7),
                     grasp_types=['side', 'top', 'wide'],
                     rotate_expansion=180)
objects['pudding_box'] = pudding_box

gelatin_box = Object(name='gelatin_box',
                     file_path=PATH + 'ycb_models/009_gelatin_box/textured.obj',
                     init_pose=(0, np.pi / 2, -np.pi * 0.58),
                     grasp_types=['side', 'top', 'wide'],
                     rotate_expansion=180)
objects['gelatin_box'] = gelatin_box

potted_meat_can = Object(name='potted_meat_can',
                         file_path=PATH + 'ycb_models/010_potted_meat_can/textured_simple.obj',
                         init_pose=(-np.pi / 2, 0, 0),
                         grasp_types=['side', 'top', 'wide'],
                         rotate_expansion=180)
objects['potted_meat_can'] = potted_meat_can

banana = Object(name='banana',
                file_path=PATH + 'ycb_models/011_banana/textured.obj',
                init_pose=(-np.pi / 2, 0, np.pi * 0.1),
                grasp_types=['side'])
objects['banana'] = banana

pitcher_base = Object(name='pitcher_base',
                      file_path=PATH + 'ycb_models/019_pitcher_base/textured_simple.obj',
                      init_pose=(-np.pi / 2, 0, -np.pi / 4),
                      grasp_types=['handle', 'top'])
objects['pitcher_base'] = pitcher_base

bleach_cleanser = Object(name='bleach_cleanser',
                         file_path=PATH + 'ycb_models/021_bleach_cleanser/textured.obj',
                         init_pose=(-np.pi / 2, 0, np.pi / 2),
                         grasp_types=['side', 'top', 'wide'],
                         rotate_expansion=180)
objects['bleach_cleanser'] = bleach_cleanser

bowl = Object(name='bowl',
              file_path=PATH + 'ycb_models/024_bowl/textured.obj',
              init_pose=(-np.pi / 2, 0, 0),
              grasp_types=['near', 'side'],
              rotate_expansion=90)
objects['bowl'] = bowl

mug = Object(name='mug',
             file_path=PATH + 'ycb_models/025_mug/textured_simple.obj',
             init_pose=(-np.pi / 2, 0, 0),
             grasp_types=['handle', 'side', 'top'],
             g_clusters=1)
objects['mug'] = mug

power_drill = Object(name='power_drill',
                     file_path=PATH + 'ycb_models/035_power_drill/textured_simple.obj',
                     init_pose=(0, 0, 0),
                     grasp_types=['handle', 'head'],
                     g_clusters=1)
objects['power_drill'] = power_drill

wood_block = Object(name='wood_block',
                    file_path=PATH + 'ycb_models/036_wood_block/textured.obj',
                    init_pose=(-np.pi / 2, 0, np.pi / 13),
                    grasp_types=['side', 'top'],
                    rotate_expansion=180)
objects['wood_block'] = wood_block

scissors = Object(name='scissors',
                  file_path=PATH + 'ycb_models/037_scissors/textured.obj',
                  init_pose=(-np.pi / 2, 0, np.pi * 0.57),
                  grasp_types=['head', 'pinch', 'wide'],
                  g_clusters=1)
objects['scissors'] = scissors

large_marker = Object(name='large_marker',
                      file_path=PATH + 'ycb_models/040_large_marker/textured.obj',
                      init_pose=(-np.pi / 2, 0, -np.pi / 2),
                      grasp_types=['handle', 'head', 'middle'])
objects['large_marker'] = large_marker

large_clamp = Object(name='large_clamp',
                     file_path=PATH + 'ycb_models/051_large_clamp/textured.obj',
                     init_pose=(-np.pi / 2, 0, -np.pi * 0.53),
                     grasp_types=['handle', 'head'])
objects['large_clamp'] = large_clamp

extra_large_clamp = Object(name='extra_large_clamp',
                           file_path=PATH + 'ycb_models/052_extra_large_clamp/textured.obj',
                           init_pose=(-np.pi / 2, 0, 0),
                           grasp_types=['handle', 'head'])
objects['extra_large_clamp'] = extra_large_clamp

foam_brick = Object(name='foam_brick',
                    file_path=PATH + 'ycb_models/061_foam_brick/textured.obj',
                    init_pose=(0, np.pi / 2, 0),
                    grasp_types=['side', 'top'],
                    rotate_expansion=180)
objects['foam_brick'] = foam_brick

if __name__ == "__main__":
    object_cls = objects['mustard_bottle']
    # Coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    # Object
    object_mesh = object_cls.init_transform()
    # object_mesh = o3d.io.read_triangle_mesh('models\ycb_models/025_mug/textured.obj', True)
    meshes = [coordinate, object_mesh]
    o3d.visualization.draw_geometries(meshes)

    # init_pose = list(object_cls.init_pose)
    # print(init_pose, type(init_pose))
    # new_pose = [item * (180/np.pi) for item in init_pose]
    # print(new_pose)

