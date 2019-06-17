# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""

A running demo of elevators

Authors: wangfan04(wangfan04@baidu.com)
Date:    2019/05/22 19:30:16
"""

import sys
import argparse
import configparser
sys.path.append('.')
sys.path.append('..')
from intrabuildingtransport.mansion.mansion_manager import MansionManager
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator

#Switch the dispatcher here
#from intrabuildingtransport.dispatchers.rule_benchmark_dispatcher import RuleBenchmarkDispatcher as Dispatcher
from intrabuildingtransport.dispatchers.rl_benchmark_dispatcher.rl_dispatcher import RLBenchmarkDispatcher as Dispatcher


def reward(time_consumption, energy_consumption, given_up_persons):
  return - (time_consumption + 0.01 * energy_consumption + 1000 * given_up_persons) * 1.0e-5

def run_mansion_main(mansion_sim, dispatcher_handle, iteration, log_level):
  print("Running simulation on Mansion: %s"%mansion_sim.name)
  mansion_sim.reset_env()
  mansion_sim._config.set_logger_level(log_level)
  dispatcher_handle.link_mansion(mansion_sim)
  dispatcher_handle.load_settings()
  i = 0
  acc_reward = 0.0
  acc_time = 0.0
  acc_energy = 0.0
  while i < iteration:
    i += 1
    state = mansion_sim.state
    # print(state)
    action = dispatcher_handle.policy(state)
    # print ("action:", action)
    time_consume, energy_consume, given_up_persons = mansion_sim.run_mansion(action)
    r = reward(time_consume, energy_consume, given_up_persons)
    dispatcher_handle.feedback(state, action, r)
    acc_reward += r
    acc_time += time_consume
    acc_energy += energy_consume
    if(i % 3600 == 0):
      print("Time Consumption: %f, Energy Consumption: %f, Accumulated Reward: %f, Mansion Status: %s"%(
        acc_time, acc_energy, acc_reward, mansion_sim.get_statistics()
        ))
      acc_time = 0.0
      acc_energy = 0.0
      acc_reward = 0.0

#run main program with args
def run_main(args):
  parser = argparse.ArgumentParser(description='Run elevator simulation')
  parser.add_argument('configfile', type=str, #default='../config.ini',
                          help='configuration file for running elevators')
  args = parser.parse_args(args)
  config = configparser.ConfigParser()
  config.read(args.configfile)

  person_generators = dict()

  time_step = float(config['General_Info']['RunningTimeStep'])
  mansion_type = config['General_Info']['MansionType']

  #Readin different person generators
  for i in range(int(config['General_Info']['PersonGeneratorNumber'])):
    key = 'Person_Generator_%d'%(i + 1)
    name = config[key]['PersonGeneratorName']
    gtype = config[key]['PersonGeneratorType']
    person_generators[name] = PersonGenerator(gtype)
    person_generators[name].configure(config[key])

  mansion_dict = {}
  for i in range(int(config['General_Info']['MansionNumber'])):
    key = 'Mansion_Info_%d'%(i + 1)
    name = config[key]['MansionName']
    gen_name = config[key]['MansionPersonGenerator']
    mansion_config = MansionConfig(
        dt = time_step, 
        number_of_floors = int(config[key]['MansionFloors']),
        floor_height = float(config[key]['MansionFloorHeight'])
        )
    mansion_dict[name] = MansionManager(
        int(config[key]['MansionElevatorNumber']),
        person_generators[gen_name],
        mansion_config,
        name)

  key = 'Mansion_Info_Fixed'
  name = config[key]['MansionName']
  gen_name = config[key]['MansionPersonGenerator']
  std_mansion_config = MansionConfig(
      dt = time_step, 
      number_of_floors = int(config[key]['MansionFloors']),
      floor_height = float(config[key]['MansionFloorHeight'])
      )
  standard_mansion = MansionManager(
        int(config[key]['MansionElevatorNumber']),
        person_generators[gen_name],
        std_mansion_config,
        name)

  iterations = float(config['General_Info']['DurationDays']) * 86400 / time_step
  dispatcher = Dispatcher()

  if(mansion_type == 'Fixed'):
    run_mansion_main(standard_mansion, dispatcher, iterations, config['General_Info']['LogLevel'])
  elif(mansion_type == 'Diversified'):
    for key in mansion_dict:
      run_mansion_main(mansion_dict[key], dispatcher, iterations, config['General_Info']['LogLevel'])
  else:
    raise Exception("No such type: %s"%mansion_type)

  return 0

if __name__ == "__main__":
  run_main(sys.argv[1:])
