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

import paddle.fluid as fluid
import parl.layers as layers
import numpy as np
from parl.framework.model_base import Model
from parl.framework.agent_base import Agent

from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler

class RLDispatcherModel(Model):
    def __init__(self, act_dim):
        self._act_dim = act_dim
        self._fc_1 = layers.fc(size=512, act='relu')
        self._fc_2 = layers.fc(size=256, act='relu')
        self._fc_3 = layers.fc(size=128, act='tanh')

        self.value_fc = layers.fc(size=1)
        self.policy_fc = layers.fc(size=act_dim)

    def policy(self, obs):
        """
        Args:obs: A float32 tensor 
        Returns:policy_logits: B * ACT_DIM
        """
        h_1 = self._fc_1(obs)
        h_2 = self._fc_2(h_1)
        h_3 = self._fc_3(h_2)
        policy_logits = self.policy_fc(h_3)
        return policy_logits

    def value(self, obs):
        """
        Args:       obs: A float32 tensor 
        Returns:    values: B
        """
        h_1 = self._fc_1(obs)
        h_2 = self._fc_2(h_1)
        h_3 = self._fc_3(h_2)
        values = self.value_fc(h_3)
        values = layers.squeeze(values, axes=[1])
        return values

    def policy_and_value(self, obs):
        """
        Args:       obs: A float32 tensor
        Returns:    policy_logits: B * ACT_DIM
                    values: B
        """
        # print('obs.shape: ', obs.shape)
        h_1 = self._fc_1(obs)
        h_2 = self._fc_2(h_1)
        h_3 = self._fc_3(h_2)
        policy_logits = self.policy_fc(h_3)
        values = self.value_fc(h_3)
        values = layers.squeeze(values, axes=[1])

        return policy_logits, values


class ElevatorAgent(Agent):
    def __init__(self, algorithm, config, learn_data_provider=None):
        self.config = config
        super(ElevatorAgent, self).__init__(algorithm)

        use_cuda = True if self.gpu_id >= 0 else False

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True

        # Use ParallelExecutor to make learn program run faster
        self.learn_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self.learn_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        if learn_data_provider:
            self.learn_reader.decorate_tensor_provider(learn_data_provider)
            self.learn_reader.start()

    def build_program(self):
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()
        self.config['obs_shape'] = [self.config['obs_shape']]

        with fluid.program_guard(self.sample_program):
            obs = layers.data(
                name='obs', shape=self.config['obs_shape'], dtype='float32')
            self.sample_actions, self.behaviour_logits = self.alg.sample(obs)

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=self.config['obs_shape'], dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=self.config['obs_shape'], dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')
            behaviour_logits = layers.data(
                name='behaviour_logits',
                shape=[self.config['act_dim']],
                dtype='float32')
            rewards = layers.data(name='rewards', shape=[], dtype='float32')
            dones = layers.data(name='dones', shape=[], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(
                name='entropy_coeff', shape=[], dtype='float32')

            self.learn_reader = fluid.layers.create_py_reader_by_data(
                capacity=32,
                feed_list=[
                    obs, actions, behaviour_logits, rewards, dones, lr,
                    entropy_coeff
                ])

            obs, actions, behaviour_logits, rewards, dones, lr, entropy_coeff = fluid.layers.read_file(
                self.learn_reader)

            vtrace_loss, kl = self.alg.learn(obs, actions, behaviour_logits,
                                             rewards, dones, lr, entropy_coeff)
            self.learn_outputs = [
                vtrace_loss.total_loss.name, vtrace_loss.pi_loss.name,
                vtrace_loss.vf_loss.name, vtrace_loss.entropy.name, kl.name
            ]

    def sample(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
            Format of image input should be NCHW format.

        Returns:
            sample_ids: a numpy int64 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        sample_actions, behaviour_logits = self.fluid_executor.run(
            self.sample_program,
            feed={'obs': obs_np},
            fetch_list=[self.sample_actions, self.behaviour_logits])
        return sample_actions, behaviour_logits

    def predict(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space)
            Format of image input should be NCHW format.

        Returns:
            sample_ids: a numpy int64 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        predict_actions = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs_np},
            fetch_list=[self.predict_actions])[0]
        return predict_actions

    def learn(self):
        total_loss, pi_loss, vf_loss, entropy, kl = self.learn_exe.run(
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, kl
