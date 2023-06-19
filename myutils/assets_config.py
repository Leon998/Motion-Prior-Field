from myutils.object_config import objects
from myutils.hand_config import *


class assets:
    def __init__(self,
                 mesh: object,
                 translation: object=(0, 0, 0),
                 rotation: object=(1, 0, 0, 0),
                 delta_translation: object=0, 
                 delta_R: object=0):
        self.mesh = mesh
        self.translation = translation
        self.rotation = rotation
        self.delta_translation = delta_translation
        self.delta_R = delta_R
    
    def update_transform(self, mesh_pose):
        last_translation = np.array(self.translation)
        self.translation = np.array(mesh_pose[4:])
        delta_translation = self.translation - last_translation
    
        last_rotation = self.rotation
        self.rotation = tuple((mesh_pose[3], mesh_pose[0], mesh_pose[1], mesh_pose[2]))
        last_R = self.mesh.get_rotation_matrix_from_quaternion(last_rotation)
        current_R = self.mesh.get_rotation_matrix_from_quaternion(self.rotation)
        delta_R = (current_R).dot(np.linalg.inv(last_R))

        self.mesh.translate(delta_translation, relative=True)
        self.mesh.rotate(delta_R, center=self.translation)