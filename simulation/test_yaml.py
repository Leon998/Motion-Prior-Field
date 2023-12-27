import yaml
import pybullet as p
from math import pi

with open('simulation/obj_init.yml', 'r', encoding='utf-8') as f:
    obj_init = yaml.load(f.read(), Loader=yaml.FullLoader)

T_wo = obj_init["mug"]["trans"]
Q_wo = obj_init["mug"]["rot"]
target_idx = obj_init["mug"]["target_idx"]
print(target_idx)

for trial in range(3):
    q_wo = Q_wo[trial]
    t_wo = T_wo[trial]
    print(q_wo, t_wo)
