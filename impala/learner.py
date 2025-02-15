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
import os
import queue
import time
import threading
from parl import RemoteManager
from parl.algorithms import IMPALA
from parl.utils import logger, CSVLogger
from parl.utils.scheduler import PiecewiseScheduler
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat

import sys
sys.path.append('./')
from intrabuildingtransport.env import IntraBuildingEnv
from utils import mansion_state_preprocessing, obs_dim, act_dim, action_idx_to_action, action_to_action_idx
from model import RLDispatcherModel, ElevatorAgent

class Learner(object):
    def __init__(self, config):
        self.config = config
        self.sample_data_queue = queue.Queue(
            maxsize=config['sample_queue_max_size'])

        #=========== Create Agent ==========
        env = IntraBuildingEnv("config.ini")
        self._mansion_attr = env._mansion.attribute
        self._obs_dim = obs_dim(self._mansion_attr)
        self._act_dim = act_dim(self._mansion_attr)

        self.config['obs_shape'] = self._obs_dim
        self.config['act_dim'] = self._act_dim

        model = RLDispatcherModel(self._act_dim)
        algorithm = IMPALA(model, hyperparas=config)
        self.agent = ElevatorAgent(algorithm, config, self.learn_data_provider)

        self.cache_params = self.agent.get_params()
        self.params_lock = threading.Lock()
        self.params_updated = False
        self.cache_params_sent_cnt = 0
        self.total_params_sync = 0

        #========== Learner ==========
        self.lr, self.entropy_coeff = None, None
        self.lr_scheduler = PiecewiseScheduler(config['lr_scheduler'])
        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.kl_stat = WindowStat(100)
        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        self.learn_thread = threading.Thread(target=self.run_learn)
        self.learn_thread.setDaemon(True)
        self.learn_thread.start()

        #========== Remote Actor ===========
        self.remote_count = 0

        self.batch_buffer = []
        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.remote_manager_thread = threading.Thread(
            target=self.run_remote_manager)
        self.remote_manager_thread.setDaemon(True)
        self.remote_manager_thread.start()

        self.csv_logger = CSVLogger(
            os.path.join(logger.get_dir(), 'result.csv'))

        from utils import Summary
        self.summary = Summary('./output')


    def learn_data_provider(self):
        """ Data generator for fluid.layers.py_reader
        """
        while True:
            sample_data = self.sample_data_queue.get()
            self.sample_total_steps += sample_data['obs'].shape[0]
            self.batch_buffer.append(sample_data)

            buffer_size = sum(
                [data['obs'].shape[0] for data in self.batch_buffer])
            if buffer_size >= self.config['train_batch_size']:
                batch = {}
                for key in self.batch_buffer[0].keys():
                    batch[key] = np.concatenate(
                        [data[key] for data in self.batch_buffer])
                self.batch_buffer = []

                obs_np = batch['obs'].astype('float32')
                actions_np = batch['actions'].astype('int64')
                behaviour_logits_np = batch['behaviour_logits'].astype(
                    'float32')
                rewards_np = batch['rewards'].astype('float32')
                dones_np = batch['dones'].astype('float32')

                self.lr = self.lr_scheduler.step()
                self.entropy_coeff = self.entropy_coeff_scheduler.step()

                yield [
                    obs_np, actions_np, behaviour_logits_np, rewards_np,
                    dones_np, self.lr, self.entropy_coeff
                ]

    def run_learn(self):
        """ Learn loop
        """
        while True:
            with self.learn_time_stat:
                total_loss, pi_loss, vf_loss, entropy, kl = self.agent.learn()

            self.params_updated = True

            self.total_loss_stat.add(total_loss)
            self.pi_loss_stat.add(pi_loss)
            self.vf_loss_stat.add(vf_loss)
            self.entropy_stat.add(entropy)
            self.kl_stat.add(kl)

    def run_remote_manager(self):
        """ Accept connection of new remote actor and start sampling of the remote actor.
        """
        remote_manager = RemoteManager(port=self.config['server_port'])
        logger.info('Waiting for remote actors connecting.')
        while True:
            remote_actor = remote_manager.get_remote()
            self.remote_count += 1
            logger.info('Remote actor count: {}'.format(self.remote_count))
            if self.start_time is None:
                self.start_time = time.time()

            remote_thread = threading.Thread(
                target=self.run_remote_sample, args=(remote_actor, ))
            remote_thread.setDaemon(True)
            remote_thread.start()

    def run_remote_sample(self, remote_actor):
        """ Sample data from remote actor and update parameters of remote actor.
        """
        cnt = 0
        remote_actor.set_params(self.cache_params)
        while True:
            batch = remote_actor.sample()
            if batch:
                self.sample_data_queue.put(batch)

            cnt += 1
            if cnt % self.config['get_remote_metrics_interval'] == 0:
                metrics = remote_actor.get_metrics()
                if metrics:
                    self.remote_metrics_queue.put(metrics)

            self.params_lock.acquire()

            if self.params_updated and self.cache_params_sent_cnt >= self.config[
                    'params_broadcast_interval']:
                self.params_updated = False
                self.cache_params = self.agent.get_params()
                self.cache_params_sent_cnt = 0
            self.cache_params_sent_cnt += 1
            self.total_params_sync += 1

            self.params_lock.release()

            remote_actor.set_params(self.cache_params)

    def log_metrics(self):
        """ Log metrics of learner and actors
        """
        if self.start_time is None:
            return

        metrics = []
        while True:
            try:
                metric = self.remote_metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break

        episode_rewards, episode_steps, episode_shaping_rewards, episode_deliver_rewards, episode_wrong_deliver_rewards = [], [], [], [], []
        for x in metrics:
            episode_rewards.extend(x['episode_rewards'])
            episode_steps.extend(x['episode_steps'])
            episode_shaping_rewards.extend(x['episode_shaping_rewards'])
            episode_deliver_rewards.extend(x['episode_deliver_rewards'])
            episode_wrong_deliver_rewards.extend(x['episode_wrong_deliver_rewards'])

        max_episode_rewards, mean_episode_rewards, min_episode_rewards, \
                max_episode_steps, mean_episode_steps, min_episode_steps =\
                None, None, None, None, None, None
        mean_episode_shaping_rewards, mean_episode_deliver_rewards, mean_episode_wrong_deliver_rewards = \
                None, None, None
        if episode_rewards:
            mean_episode_rewards = np.mean(np.array(episode_rewards).flatten())
            max_episode_rewards = np.max(np.array(episode_rewards).flatten()) 
            min_episode_rewards = np.min(np.array(episode_rewards).flatten())

            mean_episode_steps = np.mean(np.array(episode_steps).flatten())
            max_episode_steps = np.max(np.array(episode_steps).flatten())
            min_episode_steps = np.min(np.array(episode_steps).flatten())
            
            mean_episode_shaping_rewards = np.mean(np.array(episode_shaping_rewards).flatten())
            mean_episode_deliver_rewards = np.mean(np.array(episode_deliver_rewards).flatten())
            mean_episode_wrong_deliver_rewards = np.mean(np.array(episode_wrong_deliver_rewards).flatten())

        #print(self.learn_time_stat.time_samples.items)
        
        metric = {
            'Sample steps': self.sample_total_steps,
            'max_episode_rewards': max_episode_rewards,
            'mean_episode_rewards': mean_episode_rewards,
            'min_episode_rewards': min_episode_rewards,
            'mean_episode_shaping_rewards': mean_episode_shaping_rewards,
            'mean_episode_deliver_rewards': mean_episode_deliver_rewards,
            'mean_episode_wrong_deliver_rewards': mean_episode_wrong_deliver_rewards,
            #'max_episode_steps': max_episode_steps,
            #'mean_episode_steps': mean_episode_steps,
            #'min_episode_steps': min_episode_steps,
            'sample_queue_size': self.sample_data_queue.qsize(),
            'total_params_sync': self.total_params_sync,
            'cache_params_sent_cnt': self.cache_params_sent_cnt,
            'total_loss': self.total_loss_stat.mean,
            'pi_loss': self.pi_loss_stat.mean,
            'vf_loss': self.vf_loss_stat.mean,
            'entropy': self.entropy_stat.mean,
            'kl': self.kl_stat.mean,
            'learn_time_s': self.learn_time_stat.mean,
            'elapsed_time_s': int(time.time() - self.start_time),
            'lr': self.lr,
            'entropy_coeff': self.entropy_coeff,
        }

        logger.info(metric)
        self.csv_logger.log_dict(metric)
        self.summary.log_dict(metric, self.sample_total_steps)

    def close(self):
        self.csv_logger.close()
