# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""

qa test for elevators

Authors: likejiao(likejiao@baidu.com)
Date:    2019/06/16 19:30:16
"""

import sys
import time
import copy
import traceback
sys.path.append('./')

from intrabuildingtransport.env import IntraBuildingEnv
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState, ElevatorAction
from intrabuildingtransport.mansion.mansion_manager import MansionManager


fail_flag = False
stop_count = 10


def state_check(state, next_state, action):
    global fail_flag
    global stop_count
    try:
        assert isinstance(state, MansionState)
        # for e in state.ElevatorStates:
        for i in range(len(state.ElevatorStates)):
            ele = copy.deepcopy(state.ElevatorStates[i])
            assert isinstance(ele, ElevatorState)
            next_ele = copy.deepcopy(next_state.ElevatorStates[i])
            assert isinstance(next_ele, ElevatorState)
            act = copy.deepcopy(action[i])
            assert isinstance(act, ElevatorAction)
            # type
            ele_Floor = ele.Floor
            ele_Velocity = ele.Velocity
            ele_LoadWeight = ele.LoadWeight
            next_ele_Floor = next_ele.Floor
            next_ele_Velocity = next_ele.Velocity
            next_ele_LoadWeight = next_ele.LoadWeight

            assert isinstance(ele_Floor, float)
            assert isinstance(ele.MaximumFloor, int)
            assert isinstance(ele_Velocity, float)
            assert isinstance(ele.MaximumSpeed, float)
            assert isinstance(ele.Direction, int)
            assert isinstance(ele.CurrentDispatchTarget, int)
            assert isinstance(ele.DispatchTargetDirection, int)
            assert isinstance(ele_LoadWeight, float)
            assert isinstance(ele.MaximumLoad, int)
            assert isinstance(ele.OverloadedAlarm, float)
            assert isinstance(ele.DoorIsOpening, bool)
            assert isinstance(ele.DoorIsClosing, bool)
            assert isinstance(ele.ReservedTargetFloors, list)
            # change
            ele_Floor = round(ele_Floor, 2)
            ele_Velocity = round(ele_Velocity, 2)
            ele_LoadWeight = round(ele_LoadWeight, 2)
            
            # range
            assert ele_Floor > 0 and ele_Floor <= ele.MaximumFloor
            assert ele_Velocity >= (0 - ele.MaximumSpeed) and ele_Velocity <= ele.MaximumSpeed
            assert ele.Direction in [-1, 0, 1]
            assert ele.CurrentDispatchTarget >= -1 and ele.CurrentDispatchTarget <= ele.MaximumFloor
            assert ele.DispatchTargetDirection in [-1, 1]
            assert ele_LoadWeight >= 0 and ele_LoadWeight <= ele.MaximumLoad
            assert ele.OverloadedAlarm >= 0 and ele.OverloadedAlarm <= 2.0
            assert ele.DoorState >= 0 and ele.DoorState <= 1
            assert ele.DoorIsClosing in [True, False]
            assert ele.DoorIsOpening in [True, False]
            for t in ele.ReservedTargetFloors:
                assert t >= 1 and t <= ele.MaximumFloor
            
            #relation
            # if(ele_Velocity == 0 and ele.CurrentDispatchTarget != 0):
                # assert (abs(round(ele_Floor) - (ele_Floor % 1)) <= 0) or (ele_Floor % 1 != 0 and ele.Direction == 0)
            if(ele_Floor % 1 != 0 and ele.Direction != 0):
                assert ele_Velocity != 0 or next_ele_Velocity != 0 or\
                         next_ele.Direction == 0 or ele_Floor == ele.CurrentDispatchTarget
            assert (ele.DoorIsClosing and ele.DoorIsOpening) == False
            if(ele.DoorState < 1 and ele.DoorState > 0):
                assert (ele.DoorIsClosing or ele.DoorIsOpening) == True  
                assert ele_Floor % 1 == 0
            # if(ele.DoorState in [0.0, 1.0]):
            #     assert (ele.DoorIsClosing or ele.DoorIsOpening) == False  # ignore
            if(ele.DoorState in [0.0, 1.0]):
                if((ele.DoorIsClosing or ele.DoorIsOpening) == True):
                    if(next_ele.DoorState in [0.0, 1.0]):
                        assert (next_ele.DoorIsClosing or next_ele.DoorIsOpening) == False
            if((ele_Floor % 1 != 0) or ((ele.DoorIsClosing and ele.DoorIsOpening) == True)):
                assert ele.DoorState == 0.0
                assert ele.DoorIsClosing == False
                assert ele.DoorIsOpening == False
            if(ele_Velocity != 0.0 and ele.Direction != 0):
                assert ele.DoorState == 0.0
            if(ele.OverloadedAlarm > 0):
                assert ele_LoadWeight >= ele.MaximumLoad - 100
            if(len(ele.ReservedTargetFloors) != 0):
                assert ele_LoadWeight >= 20

            # # dynamic check
            # delta_Floor = next_ele_Floor - ele_Floor
            # assert delta_Floor * next_ele_Velocity >= 0 or delta_Floor * ele_Velocity >= 0
            # target_list = ele.ReservedTargetFloors[:]
            # # if(ele.CurrentDispatchTarget != 0):
            # #     target_list.append(ele.CurrentDispatchTarget)
            # if(delta_Floor > 0): # going up
            #     min_target = min(target_list) if len(target_list) > 0 else ele.MaximumFloor + 1
            #     assert ele_Floor <= min_target
            #     assert next_ele_Velocity > 0 or ele_Velocity > 0
            # if(delta_Floor < 0): # going down
            #     max_target = max(target_list) if len(target_list) > 0 else 0
            #     assert ele_Floor >= max_target
            #     assert next_ele_Velocity < 0 or ele_Velocity < 0
            # if(delta_Floor == 0):
            #     assert next_ele_Velocity == 0
            
            # # if((next_ele_LoadWeight - ele_LoadWeight) > 0):
            # #     # assert len(next_ele.ReservedTargetFloors) > len(ele.ReservedTargetFloors)
            # #     assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
            # # if((next_ele_LoadWeight - ele_LoadWeight) < 0):
            # #     # assert len(next_ele.ReservedTargetFloors) < len(ele.ReservedTargetFloors)
            # #     assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
            # if(len(next_ele.ReservedTargetFloors) > len(ele.ReservedTargetFloors)):
            #     assert (next_ele_LoadWeight - ele_LoadWeight) > 0
            #     assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
            # if(len(next_ele.ReservedTargetFloors) < len(ele.ReservedTargetFloors)):
            #     # assert (next_ele_LoadWeight - ele_LoadWeight) < 0
            #     assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing

            # if(ele.OverloadedAlarm > 0):
            #     assert ele.ReservedTargetFloors == next_ele.ReservedTargetFloors
            #     assert ele_LoadWeight == next_ele_LoadWeight
            #     assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing



        if(fail_flag):
            stop_count -= 1
            if(stop_count == 0):
                print('\n\nSome error appear before several steps, please check\n\n')
                exit(1)

    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        print('An error occurred on line {} in statement {}'.format(line, text))
        print('\n========================== ele num: ', i)
        print('\nlast: ', ele)
        print('\nthis: ', next_ele)
        print('\n========================== please check\n\n')
        fail_flag = True
        # print('==========================next state')
        # print_next_state(next_state)
        # exit(1)

def print_state(state, action):
    assert isinstance(state, MansionState)
    print('Num\tact\tact.dir\tFloor\t\tMaxF\tV\t\tMaxV\tDir\tTarget\tTDir\tLoad\tMaxL\tOver\tDoor\topening\tclosing\tReservedTargetFloors')
    i = 0
    for i in range(len(state.ElevatorStates)):
        ele = state.ElevatorStates[i]
        act = action[i]
        assert isinstance(ele, ElevatorState)
        assert isinstance(act, ElevatorAction)
        print(i,"\t|",act.TargetFloor,"\t|",act.DirectionIndicator,"\t|",
                    '%2.4f'%ele.Floor,"\t|",ele.MaximumFloor,"\t|",
                    '%2.7f'%ele.Velocity,"\t|",ele.MaximumSpeed,"\t|",
                    ele.Direction,"\t|",ele.CurrentDispatchTarget,"\t|",ele.DispatchTargetDirection,"\t|",
                    int(ele.LoadWeight),"\t|",ele.MaximumLoad,"\t|",'%.2f'%ele.OverloadedAlarm,"\t|",
                    ele.DoorState,"\t|",int(ele.DoorIsOpening),"\t|",int(ele.DoorIsClosing),"\t|",ele.ReservedTargetFloors)
        i += 1
    print('------------------RequiringUpwardFloors', state.RequiringUpwardFloors)
    print('------------------RequiringDownwardFloors', state.RequiringDownwardFloors)
    print('')
    # time.sleep(2)

def print_next_state(state):
    assert isinstance(state, MansionState)
    print('Num\tact\tact.dir\tFloor\t\tMaxF\tV\tMaxV\tDir\tTarget\tTDir\tLoad\tMaxL\tOver\tDoor\topening\tclosing\tRT')
    i = 0
    for i in range(len(state.ElevatorStates)):
        ele = state.ElevatorStates[i]
        # act = action[i]
        assert isinstance(ele, ElevatorState)
        # assert isinstance(act, ElevatorAction)
        i += 1
        print(i,"\t|",' ',"\t|",' ',"\t|",
                    '%.2f'%ele.Floor,"\t|",ele.MaximumFloor,"\t|",
                    '%.1f'%ele.Velocity,"\t|",ele.MaximumSpeed,"\t|",
                    ele.Direction,"\t|",ele.CurrentDispatchTarget,"\t|",ele.DispatchTargetDirection,"\t|",
                    '%.1f'%ele.LoadWeight,"\t|",ele.MaximumLoad,"\t|",ele.OverloadedAlarm,"\t|",
                    ele.DoorState,"\t|",int(ele.DoorIsOpening),"\t|",int(ele.DoorIsClosing),"\t|",ele.ReservedTargetFloors)
    print('------------------RequiringUpwardFloors', state.RequiringUpwardFloors)
    print('------------------RequiringDownwardFloors', state.RequiringDownwardFloors)
    print('')
    # time.sleep(2)




def run_mansion_main(mansion_env, policy_handle, iteration):
    mansion_env.reset()
    policy_handle.link_mansion(mansion_env.attribute)
    policy_handle.load_settings()
    i = 0
    acc_reward = 0.0

    last_state = copy.deepcopy(mansion_env.state)

    while i < iteration:
        i += 1
        state = mansion_env.state

        action = policy_handle.policy(state)
        _, r, _ = mansion_env.step(action)
        output_info = policy_handle.feedback(state, action, r)
        acc_reward += r

        if(isinstance(output_info, dict) and len(output_info) > 0):
            mansion_env.log_notice("%s", output_info)
        if(i % 3600 == 0):
            mansion_env.log_notice(
                "Accumulated Reward: %f, Mansion Status: %s",
                acc_reward, mansion_env.statistics)
            acc_reward = 0.0

        print_state(state, action)
        state_check(last_state, state, action)
        last_state = copy.deepcopy(state)


# run main program with args
def run_qa_test(configfile, iterations, controlpolicy):
    print('configfile:', configfile) # configuration file for running elevators
    print('iterations:', iterations) # total number of iterations
    print('controlpolicy:', controlpolicy) # policy type: rule_benchmark or others

    control_module = ("dispatchers.{}.dispatcher"
                      .format(controlpolicy))
    Dispatcher = __import__(control_module, fromlist=[None]).Dispatcher

    mansion_env = IntraBuildingEnv(configfile)
    dispatcher = Dispatcher()

    run_mansion_main(mansion_env, dispatcher, iterations)

    return 0


if __name__ == "__main__":
    run_qa_test('config.ini', 100, 'rule_benchmark')
    run_qa_test('tests/conf/config1.ini', 100, 'rule_benchmark')
    run_qa_test('tests/conf/config2.ini', 100, 'rule_benchmark')
    run_qa_test('tests/conf/config3.ini', 100, 'rule_benchmark')

