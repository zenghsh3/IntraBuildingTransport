# basic_generators.py
# Author: Fan Wang (wang.fan@baidu.com)
#
# Generating Persons In a uniform manner

import sys
import random
from intrabuildingtransport.mansion.utils import EPSILON
from intrabuildingtransport.mansion.utils import PersonType
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.person_generators.person_generator import PersonGeneratorBase


class UniformPersonGenerator(PersonGeneratorBase):
    '''
    Basic Generator Class
    Generates Random Person from Random Floor, going to random other floor
    Uniform distribution in floor number, target floor number etc
    '''

    def configure(self, configuration):
        self._particle_number = int(configuration['ParticleNumber'])
        self._particle_interval = float(configuration['GenerationInterval'])
        self._cur_id = 0

    def _random_weight(self):
        MIN_WEIGHT = 20
        MAX_WEIGHT = 100
        return random.uniform(MIN_WEIGHT, MAX_WEIGHT)

    def generate_person(self):
        '''
        Generate Random Persons from Poisson Distribution
        Args:
          None
        Returns:
          List of Random Persons
        '''
        ret_persons = []
        time_interval = self._config.raw_time - self._last_generate_time
        for i in range(self._particle_number):
            if(random.random() < time_interval / self._particle_interval):
                random_source_floor = random.randint(1, self._floor_number)
                random_target_floor = random.randint(1, self._floor_number)
                while random_source_floor == random_target_floor:
                    random_source_floor = random.randint(1, self._floor_number)
                    random_target_floor = random.randint(1, self._floor_number)
                random_weight = self._random_weight()

                ret_persons.append(
                    PersonType(
                        self._cur_id,
                        random_weight,
                        random_source_floor,
                        random_target_floor,
                        self._config.raw_time))
                self._cur_id += 1
        self._last_generate_time = self._config.raw_time
        return ret_persons
