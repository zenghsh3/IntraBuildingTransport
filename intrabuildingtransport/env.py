from intrabuildingtransport.mansion.person_generators.generator_proxy import set_seed
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState
from intrabuildingtransport.mansion.mansion_manager import MansionManager
import configparser
import random
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')

import copy


class IntraBuildingEnv():
    '''
    IntraBuildingTransportation Environment
    '''

    def __init__(self, file_name):
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])

        # Readin different person generators
        gtype = config['PersonGenerator']['PersonGeneratorType']
        person_generator = PersonGenerator(gtype)
        person_generator.configure(config['PersonGenerator'])

        self._config = MansionConfig(
            dt=time_step,
            number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            floor_height=float(config['MansionInfo']['FloorHeight'])
        )

        if('LogLevel' in config['Configuration']):
            self._config.set_logger_level(config['Configuration']['LogLevel'])
        if('Lognorm' in config['Configuration']):
            self._config.set_std_logfile(config['Configuration']['Lognorm'])
        if('Logerr' in config['Configuration']):
            self._config.set_err_logfile(config['Configuration']['Logerr'])

        self._mansion = MansionManager(
            int(config['MansionInfo']['ElevatorNumber']),
            person_generator,
            self._config,
            config['MansionInfo']['Name']
        )

    def seed(self, seed=None):
        set_seed(seed)

    def step(self, action):
        time_consume, energy_consume, given_up_persons, info = self._mansion.run_mansion(
            action)
        reward = - (time_consume + 0.01 * energy_consume +
                    1000 * given_up_persons) * 1.0e-5
        
        info = {'energy_consume_reward': np.array(info['each_energy_consume']) * -0.01,
                'time_consume_reward': time_consume * -1,
                'given_up_reward': given_up_persons * -1000}
        return (self._mansion.state, reward, False, info)

    def reset(self):
        self._mansion.reset_env()
        return self._mansion.state

    def render(self):
        raise NotImplementedError()

    def close(self):
        pass

    @property
    def attribute(self):
        return self._mansion.attribute

    @property
    def state(self):
        return self._mansion.state

    @property
    def statistics(self):
        return self._mansion.get_statistics()

    @property
    def log_debug(self):
        return self._config.log_debug

    @property
    def log_notice(self):
        return self._config.log_notice

    @property
    def log_warning(self):
        return self._config.log_warning

    @property
    def log_fatal(self):
        return self._config.log_fatal

class RewardShapingWrapper(object):
    def __init__(self, env):
        self.env = env
        self.last_obs = None
        self.elevator_last_opening_floor = None
        self.elevator_num = None

    def reset(self):
        obs = self.env.reset()
        self.elevator_last_opening_floor = [None] * len(obs.ElevatorStates)
        for i in range(len(obs.ElevatorStates)):
            if obs.ElevatorStates[i].DoorIsOpening:
                self.elevator_last_opening_floor[i] = int(obs.ElevatorStates[i].Floor)

        self.last_obs = copy.deepcopy(obs)
        self.elevator_num = len(obs.ElevatorStates)
        return obs
    
    def step(self, action):
        assert self.last_obs is not None
        (obs, reward, done, info) = self.env.step(action)
        
        deliver_reward = np.array([0.0] * self.elevator_num)
        wrong_deliver_reward = np.array([0.0] * self.elevator_num)

        cur_up_required = set(obs.RequiringUpwardFloors)
        last_up_required = set(self.last_obs.RequiringUpwardFloors)
        common_up_required = cur_up_required & last_up_required
        satisfied_floors = last_up_required - common_up_required
        assert len(satisfied_floors) <= self.elevator_num
        for x in satisfied_floors:
            for i in range(self.elevator_num):
                if obs.ElevatorStates[i].DoorState == 1.0 and obs.ElevatorStates[i].Direction == 1:
                    deliver_reward[i] += 1

        cur_down_required = set(obs.RequiringDownwardFloors)
        last_down_required = set(self.last_obs.RequiringDownwardFloors)
        common_down_required = cur_down_required & last_down_required
        satisfied_floors = last_down_required - common_down_required
        assert len(satisfied_floors) <= self.elevator_num
        for x in satisfied_floors:
            for i in range(self.elevator_num):
                if obs.ElevatorStates[i].DoorState == 1.0 and obs.ElevatorStates[i].Direction == -1:
                    deliver_reward[i] += 1

        for i in range(len(obs.ElevatorStates)):
            if obs.ElevatorStates[i].DoorIsOpening:
                cur_floor =int(obs.ElevatorStates[i].Floor)
                if cur_floor != self.elevator_last_opening_floor[i]:
                    if not(cur_floor in obs.ElevatorStates[i].ReservedTargetFloors or \
                            cur_floor in cur_up_required or\
                            cur_floor in cur_down_required):
                        wrong_deliver_reward[i] -=  1.0
                self.elevator_last_opening_floor[i] = int(obs.ElevatorStates[i].Floor)

        self.last_obs = copy.deepcopy(obs)

        origin_reward = info['energy_consume_reward'] + info['time_consume_reward'] / self.elevator_num + info['given_up_reward'] / self.elevator_num

        shaping_reward = deliver_reward + wrong_deliver_reward + origin_reward * 0.001

        info['reward'] = reward 
        info['shaping_reward'] = shaping_reward
        info['deliver_reward'] = deliver_reward
        info['wrong_deliver_reward'] = wrong_deliver_reward
        info['each_origin_reward'] = origin_reward
        #print(obs, reward, done, info)
        return (obs, reward, done, info)

    @property
    def attribute(self):
        return self.env.attribute
    
    @property
    def state(self):
        return self.env.state

    @property
    def statistics(self):
        return self.env.statistics

    @property
    def log_debug(self):
        return self.env.log_debug

    @property
    def log_notice(self):
        return self.env.log_notice

    @property
    def log_warning(self):
        return self.env.log_warning

    @property
    def log_fatal(self):
        return self.env.log_fatal

    @property
    def _mansion(self):
        return self.env._mansion
