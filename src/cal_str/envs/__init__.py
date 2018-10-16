# envs:
from cal_str.envs.ball import BallEnv
from cal_str.envs.ball_ob import BallObEnv
from cal_str.envs.driving import DrivEnv
# registry:
from gym.envs.registration import registry, register, make

register(
    id = 'Ball-v0', 
    entry_point = 'cal_str.envs:BallEnv',
)

register(
    id = 'Ball-v1',
    entry_point = 'cal_str.envs:BallObEnv',
)

register(
    id = 'Driv-v0',
    entry_point = 'cal_str.envs:DrivEnv',
)
