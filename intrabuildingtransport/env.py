import sys
import argparse
import configparser
sys.path.append('.')
sys.path.append('..')
import random
from intrabuildingtransport.mansion.mansion_manager import MansionManager
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator
from intrabuildingtransport.mansion.person_generators.generator_proxy import set_seed


class IntraBuildingEnv():
  '''
  IntraBuildingTransportation Environment
  '''
  def __init__(self, file_name):
    config = configparser.ConfigParser()
    config.read(file_name)

    time_step = float(config['Configuration']['RunningTimeStep'])

    #Readin different person generators
    gtype = config['PersonGenerator']['PersonGeneratorType']
    person_generator = PersonGenerator(gtype)
    person_generator.configure(config['PersonGenerator'])

    self._config = MansionConfig(
      dt = time_step, 
      number_of_floors = int(config['MansionInfo']['NumberOfFloors']),
      floor_height = float(config['MansionInfo']['FloorHeight'])
      )

    self._config.set_logger_level(config['Configuration']['LogLevel'])
    
    self._mansion = MansionManager(
      int(config['MansionInfo']['ElevatorNumber']),
      person_generator,
      self._config,
      config['MansionInfo']['Name']
      )

  def seed(self, seed = None):
    set_seed(seed)

  def step(self, action):
    time_consume, energy_consume, given_up_persons = self._mansion.run_mansion(action)
    reward = - (time_consume + 0.01 * energy_consume + 1000 * given_up_persons) * 1.0e-5
    return (self._mansion.state, reward, {})

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
