#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import numpy as np
import parl
import six
from collections import defaultdict
from parl.algorithms import IMPALA

import sys
sys.path.append('./')
from intrabuildingtransport.env import IntraBuildingEnv, RewardShapingWrapper
from utils import mansion_state_preprocessing, obs_dim, act_dim
from utils import action_idx_to_action_batch

from model import RLDispatcherModel, ElevatorAgent
from impala_config import config

class CompuEnv(object):
    """ vector of envs to support vector reset and vector step.
    `vector_step` api will automatically reset envs which are done.
    """

    def __init__(self, env):
        """
        Args:       envs: List of env
        """
        assert isinstance(env, IntraBuildingEnv) or isinstance(env, RewardShapingWrapper)
        self.env = env
        self._mansion = env._mansion

        self._current_reward = None
        self._shaping_reward = None
        self._deliver_reward = None
        self._wrong_deliver_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_shaping_rewards = []
        self._episode_deliver_rewards = []
        self._episode_wrong_deliver_rewards = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self):
        """
        Returns:    List of all elevators' obs
        """
        state = self.env.reset()
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0
        self._shaping_reward = 0
        self._deliver_reward = 0
        self._wrong_deliver_reward = 0

        return state

    def step(self, action):
        """
        Args:       actions: List or array of action
        Returns:       obs_batch: List of next obs of envs
                    reward_batch: List of return reward of envs 
                    done_batch: List of done of envs 
        """
        # reset data after every 3600 s
        if self._total_steps % 3600 == 0:
            if self._current_reward is not None:
                self._episode_rewards.append(self._current_reward)
                self._episode_lengths.append(self._num_steps)
                self._episode_shaping_rewards.append(self._shaping_reward)
                self._episode_deliver_rewards.append(self._deliver_reward)
                self._episode_wrong_deliver_rewards.append(self._wrong_deliver_reward)
                self._num_episodes += 1
            print('%f,%d'%(self._current_reward, self._total_steps))
            self._current_reward = 0
            self._num_steps = 0
            self._shaping_reward = 0
            self._deliver_reward = 0
            self._wrong_deliver_reward = 0

        state, reward, done, info = self.env.step(action)

        self._current_reward += info['reward']
        self._shaping_reward += np.sum(info['shaping_reward'])
        self._deliver_reward += np.sum(info['deliver_reward'])
        self._wrong_deliver_reward += np.sum(info['wrong_deliver_reward'])

        self._num_steps += 1
        self._total_steps += 1
        
        return (state, reward, done, info)

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i], self._episode_shaping_rewards[i],
                    self._episode_deliver_rewards[i], self._episode_wrong_deliver_rewards[i])
        self._num_returned = len(self._episode_rewards)

class MultiVectorEnv(object):
    """ vector of envs to support vector reset and vector step.
    `vector_step` api will automatically reset envs which are done.
    """

    def __init__(self, envs, ele_num, act_dim):
        """
        Args:       envs: List of env
        """
        self.envs = envs
        self.envs_num = len(envs)
        self.ele_num = ele_num
        self.act_dim = act_dim

    def reset(self):
        """
        Returns:    List of all elevators' obs
        """
        reset_obs_batch = []
        for env in self.envs:
            obs = env.reset()
            obs_array = mansion_state_preprocessing(obs)
            reset_obs_batch.extend(obs_array)
        return reset_obs_batch

    def step(self, actions):
        """
        Args:       actions: List or array of action
        Returns:       obs_batch: List of next obs of envs
                    reward_batch: List of return reward of envs 
                    done_batch: List of done of envs 
        """
        obs_batch, reward_batch, done_batch, info_batch = [], [], [], []
        multi_actions = np.array_split(actions, int(len(actions) / self.ele_num))
        # print('multi_actions', multi_actions)
        #
        assert len(multi_actions[0]) == self.ele_num
        actions = action_idx_to_action_batch(multi_actions, self.act_dim)
        # print('actions', actions)
        #
        for env_id in six.moves.range(self.envs_num):
            obs, reward, done, info = self.envs[env_id].step(actions[env_id])
            obs_array = mansion_state_preprocessing(obs)

            if done:
                obs = self.envs[env_id].reset()
                obs_array = mansion_state_preprocessing(obs)

            obs_batch.extend(obs_array)
            reward_batch.extend([float(info['shaping_reward'][i]) for i in range(self.ele_num)])
            done_batch.extend([done for i in range(self.ele_num)])
            info_batch.extend([info for i in range(self.ele_num)])
        return obs_batch, reward_batch, done_batch, info_batch



@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config

        self.envs = []
        for _ in six.moves.range(config['env_num']):
            env = IntraBuildingEnv("config.ini")
            env = RewardShapingWrapper(env)
            env = CompuEnv(env)
            self.envs.append(env)

        self._mansion_attr = env._mansion.attribute
        self._obs_dim = obs_dim(self._mansion_attr)
        self._act_dim = act_dim(self._mansion_attr)

        self.config['obs_shape'] = self._obs_dim
        self.config['act_dim'] = self._act_dim

        self.ele_num = self._mansion_attr.ElevatorNumber
        self.max_floor = self._mansion_attr.NumberOfFloor 
        self.config['ele_num'] = self.ele_num
        self.config['max_floor'] = self.max_floor

        self.vector_env = MultiVectorEnv(self.envs, self.ele_num, self._act_dim)

        self.obs_batch = self.vector_env.reset()

        model = RLDispatcherModel(self._act_dim)
        algorithm = IMPALA(model, hyperparas=config)
        self.agent = ElevatorAgent(algorithm, config)

    def sample(self):
        env_sample_data = {}
        for env_id in six.moves.range(self.config['env_num'] * self.ele_num):
            env_sample_data[env_id] = defaultdict(list)

        for i in six.moves.range(self.config['sample_batch_steps']):
            actions, behaviour_logits = self.agent.sample(
                np.stack(self.obs_batch))
            next_obs_batch, reward_batch, done_batch, info_batch = \
                    self.vector_env.step(actions)

            for env_id in six.moves.range(self.config['env_num'] * self.ele_num):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(actions[env_id])
                env_sample_data[env_id]['behaviour_logits'].append(
                    behaviour_logits[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])

            self.obs_batch = next_obs_batch

        # Merge data of envs
        sample_data = defaultdict(list)
        for env_id in six.moves.range(self.config['env_num']):
            for data_name in [
                    'obs', 'actions', 'behaviour_logits', 'rewards', 'dones'
            ]:
                sample_data[data_name].extend(
                    env_sample_data[env_id][data_name])

        # size of sample_data: env_num * sample_batch_steps
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

    def get_metrics(self):
        metrics = defaultdict(list)
        for env in self.envs:
            monitor = env
            if monitor is not None:
                for episode_rewards, episode_steps, episode_shaping_rewards, episode_deliver_rewards, episode_wrong_deliver_rewards in monitor.next_episode_results(
                ):
                    metrics['episode_rewards'].append(episode_rewards)
                    metrics['episode_steps'].append(episode_steps)
                    metrics['episode_shaping_rewards'].append(episode_shaping_rewards)
                    metrics['episode_deliver_rewards'].append(episode_deliver_rewards)
                    metrics['episode_wrong_deliver_rewards'].append(episode_wrong_deliver_rewards)
        return metrics

    def set_params(self, params):
        self.agent.set_params(params)


if __name__ == '__main__':
    from impala_config import config

    actor = Actor(config)
    actor.as_remote(config['server_ip'], config['server_port'])
