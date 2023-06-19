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