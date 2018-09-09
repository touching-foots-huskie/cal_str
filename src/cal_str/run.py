import os
import cal_str
import mujoco_py
from mpi4py import MPI
from baselines import logger
from baselines.common import tf_util as U
from baselines.common import set_global_seeds
from cal_str.network import cal_policy, pposgd_cal
from cal_str.network.knowledge import knowledge_dict
from cal_str.network.knowledge import network_config

from baselines.bench import Monitor

# get environment:
def make_mujoco_env(env_id, seed, reward_scale=1.0):
    '''
    Create a wrapped, monitered gym.Env for MuJoCo
    '''
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = cal_str.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env 

def train(config, env_id, num_timesteps, seed):
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, config, ob_space, ac_space):
        return cal_policy.CalPolicy(name=name, config=config, ob_space=ob_space, ac_space=ac_space)

    env = make_mujoco_env(env_id, seed)
    pposgd_cal.learn(config, env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def gen_config():
    # only using config
    config = dict()
    config['run_type'] = 'train'
    config['continue'] = False
    # construction configuration:
    config['env_type'] = 'reacher'
    config['update_name'] = 'reacher'
    # network config:
    network_config(config)
    return config


def main():
    logger.configure()
    config = gen_config()
    train(config, config['environment'], num_timesteps=1e6, seed=0)


if __name__ == '__main__':
    main()
