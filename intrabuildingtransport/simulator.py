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


#Switch the dispatcher here
from intrabuildingtransport.dispatchers.rule_benchmark_dispatcher import RuleBenchmarkDispatcher as Dispatcher
#from intrabuildingtransport.dispatchers.rl_benchmark_dispatcher.rl_dispatcher import RLBenchmarkDispatcher as Dispatcher


class Simulator():
    def __init__(self, file):
        # parser = argparse.ArgumentParser(description='Run elevator simulation')
        # parser.add_argument('configfile', type=str, default = '../config.ini',
        #                         help='configuration file for running elevators')
        # args = parser.parse_args()
        config = configparser.ConfigParser()
        config.read(file)

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
        
        self.mansion_dict = {}
        for i in range(int(config['General_Info']['MansionNumber'])):
            key = 'Mansion_Info_%d'%(i + 1)
            name = config[key]['MansionName']
            gen_name = config[key]['MansionPersonGenerator']
            mansion_config = MansionConfig(
                dt = time_step, 
                number_of_floors = int(config[key]['MansionFloors']),
                floor_height = float(config[key]['MansionFloorHeight'])
                )
            self.mansion_dict[name] = MansionManager(
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
        # TODO: is iteration needed in environment?
        self.iterations = float(config['General_Info']['DurationDays']) * 86400 / time_step

        if(mansion_type == 'Fixed'):
            self.mansion_sim = standard_mansion
            self.state = self.mansion_sim.reset_env()
            self.mansion_sim._config.set_logger_level(config['General_Info']['LogLevel'])
        # elif(mansion_type == 'Diversified'):
        #     pass #TODO:
        else:
            raise Exception("No such type: %s"%mansion_type)
    
    def seed(self, seed = None):
        set_seed(seed)

    def step(self, action):
        time_consume, energy_consume, given_up_persons = self.mansion_sim.run_mansion(action)
        reward = - (time_consume + 0.01 * energy_consume + 1000 * given_up_persons) * 1.0e-5
        self.state = self.mansion_sim.state
        return self.state, reward, {}

    def reset(self):
        self.mansion_sim.reset_env()
        self.state = self.mansion_sim.state 
        return self.state   #TODO: add current time???

    def render(self):
        pass

    def close(self):
        pass