import os
import pdb
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class BallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.geoms = []
        self.geoms_size = []
        self.concerned_geom = ['target']
        # mujoco structure
        file_path = os.path.join(os.path.dirname(__file__), 'assets', 'ball.xml')
        pdb.set_trace()
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        #  exam geoms:
        if not self.geoms:
            self.geoms = DynamicDict(self.model.geom_names, self.sim.data.geom_xpos)
            self.geoms_size = DynamicDict(self.model.geom_names, self.model.geom_size)

        self.do_simulation(a, self.frame_skip)
        # reward:
        reward = self._get_reward()
        # observation:
        ob = self._get_obs()
        # done:
        done = self._get_done()
        return ob, reward, done, {}

    def _get_obs(self):
        geom_info = np.concatenate([self.geoms[name] for name in self.concerned_geom])
        if self.sim.data.sensordata is not None:
            output = np.concatenate([self.data.qpos.flat, self.data.qvel.flat,
                                     self.sim.data.sensordata])
        else:
            output = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        if geom_info != np.array([]):
            output = np.concatenate([output, geom_info])

        return output

    def _get_reward(self):
        ball_to_target = np.linalg.norm(self.geoms['ball'] - self.geoms['target'])
        threshold = self.geoms_size['target'][0] * 1.25
        if ball_to_target <= threshold:
            reward = 20.0
        else:
            reward = 0.0
        return reward

    def _get_done(self):
        ball_to_target = np.linalg.norm(self.geoms['ball'] - self.geoms['target'])
        threshold = self.geoms_size['target'][0] * 1.25
        if ball_to_target <= threshold:
            done = True
        else:
            done = False
        return done

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0


class DynamicDict:
    def __init__(self, name_list, value_list):
        self.name_list = name_list
        self.value_list = value_list

    def __getitem__(self, name):
        _index = self.name_list.index(name)
        return self.value_list[_index]

    def __setitem__(self, name, value):
        _index = self.name_list.index(name)
        self.value_list[_index] = value

    def names(self):
        return self.name_list

    def values(self):
        return self.value_list


if __name__ == "__main__":
    env = BallEnv()
