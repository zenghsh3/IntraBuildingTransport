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
from intrabuildingtransport.mansion.mansion_manager import MansionManager
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator
from intrabuildingtransport.env import IntraBuildingEnv


#Switch the dispatcher here
#from intrabuildingtransport.dispatchers.rule_benchmark_dispatcher import RuleBenchmarkDispatcher as Dispatcher


#def reward(time_consumption, energy_consumption, given_up_persons):
#  return - (time_consumption + 0.01 * energy_consumption + 1000 * given_up_persons) * 1.0e-5

def run_mansion_main(mansion_env, policy_handle, iteration):
  mansion_env.reset()
  policy_handle.link_mansion(mansion_env.attribute)
  policy_handle.load_settings()
  i = 0
  acc_reward = 0.0
  #acc_time = 0.0
  #acc_energy = 0.0
  while i < iteration:
    i += 1
    state = mansion_env.state
    action = policy_handle.policy(state)
    _, r, _ = mansion_env.step(action)
    policy_handle.feedback(state, action, r)
    acc_reward += r
    #acc_time += time_consume
    #acc_energy += energy_consume

    if(i % 3600 == 0):
      print("Accumulated Reward: %f, Mansion Status: %s"%(acc_reward, mansion_env.statistics))
      #acc_time = 0.0
      #acc_energy = 0.0
      acc_reward = 0.0

#run main program with args
def run_main(args):
  parser = argparse.ArgumentParser(description='Run elevator simulation')
  parser.add_argument('--configfile', type=str, #default='../config.ini',
                          help='configuration file for running elevators')
  parser.add_argument('--iterations', type=int, default=100000000,
                          help='total number of iterations')
  parser.add_argument('--controlpolicy', type=str, default='rule_benchmark',
                          help='policy type: rule_benchmark or others')
  args = parser.parse_args(args)
  print('configfile:', args.configfile)
  print('iterations:', args.iterations)
  print('controlpolicy:', args.controlpolicy)

  control_module = ("dispatchers.{}.dispatcher"
      .format(args.controlpolicy))
  Dispatcher = __import__(control_module, fromlist=[None]).Dispatcher

  mansion_env = IntraBuildingEnv(args.configfile)
  dispatcher = Dispatcher()
  run_mansion_main(mansion_env, dispatcher, args.iterations)

  return 0

if __name__ == "__main__":
  run_main(sys.argv[1:])
