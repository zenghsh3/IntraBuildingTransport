import sys
import random
import queue
from intrabuildingtransport.mansion.utils import ElevatorState, ElevatorAction, MansionState
from intrabuildingtransport.mansion.utils import EPSILON, HUGE
from dispatchers.dispatcher_base import DispatcherBase


class Dispatcher(DispatcherBase):
    '''
    A rule benchmark demonstration of the dispatcher
    A dispatcher must provide policy and feedback function
    The policy function receives MansionState and output ElevatorAction Lists
    The feedback function receives reward
    '''

    def policy(self, state):
        get_up_lift = []
        get_stop_lift = []
        get_down_lift = []
        ele_is_stopped = []
        ret_actions = [
            ElevatorAction(
                0, 1) for i in range(
                self._mansion_attr.ElevatorNumber)]

        idle_ele_queue = queue.Queue()
        upward_floor_address_dict = dict()
        downward_floor_address_dict = dict()

        for i in range(len(state.ElevatorStates)):
            idle_ele_queue.put(i)

        for floor in state.RequiringUpwardFloors:
            # Addressing Elevator, Priority
            upward_floor_address_dict[floor] = (-1, -HUGE)

        for floor in state.RequiringDownwardFloors:
            downward_floor_address_dict[floor] = (-1, -HUGE)

        while not idle_ele_queue.empty():
            sel_ele = idle_ele_queue.get()
            if(state.ElevatorStates[sel_ele].Direction > 0):
                assigned = False
                sel_priority = -HUGE
                sel_floor = -1
                for upward_floor in state.RequiringUpwardFloors:
                    if(upward_floor < state.ElevatorStates[sel_ele].Floor - EPSILON):
                        continue
                    priority = state.ElevatorStates[sel_ele].Floor - \
                        upward_floor
                    if(upward_floor in state.ElevatorStates[sel_ele].ReservedTargetFloors):
                        priority = min(0.0, priority + 5.0)
                    if (state.ElevatorStates[sel_ele].Velocity < EPSILON):
                        priority -= 5.0
                    if(priority > upward_floor_address_dict[upward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_floor = upward_floor

                if(sel_floor > 0):
                    ret_actions[sel_ele] = ElevatorAction(sel_floor, 1)
                    if(upward_floor_address_dict[sel_floor][0] >= 0):
                        ret_actions[upward_floor_address_dict[sel_floor]
                                    [0]] = ElevatorAction(0, 1)
                        idle_ele_queue.put(
                            upward_floor_address_dict[sel_floor][0])
                    upward_floor_address_dict[sel_floor] = (
                        sel_ele, sel_priority)
                    assigned = True

                # In case no floor is assigned to the current elevator, we
                # search all requiring downward floor, find the largest floor
                # and assign it to the elevator
                if(not assigned):
                    if(len(state.RequiringDownwardFloors) > 0):
                        max_unassigned_down_floor = -1
                        for downward_floor in state.RequiringDownwardFloors:
                            if(downward_floor_address_dict[downward_floor][0] < 0 and max_unassigned_down_floor < downward_floor):
                                max_unassigned_down_floor = downward_floor
                        if(max_unassigned_down_floor >= 0):
                            ret_actions[sel_ele] = ElevatorAction(
                                max_unassigned_down_floor, -1)
                            priority = - \
                                state.ElevatorStates[sel_ele].Floor - EPSILON + max_unassigned_down_floor
                            downward_floor_address_dict[max_unassigned_down_floor] = (
                                sel_ele, priority)

                # print (sel_ele, "going up, sel floor", ret_actions[sel_ele], sel_priority)

            if(state.ElevatorStates[sel_ele].Direction < 0):
                assigned = False
                sel_priority = -HUGE
                sel_floor = -1
                for downward_floor in state.RequiringDownwardFloors:
                    if(downward_floor > state.ElevatorStates[sel_ele].Floor + EPSILON):
                        continue
                    priority = - \
                        state.ElevatorStates[sel_ele].Floor + downward_floor
                    if(downward_floor in state.ElevatorStates[sel_ele].ReservedTargetFloors):
                        priority = min(0.0, priority + 5.0)
                    if (state.ElevatorStates[sel_ele].Velocity < EPSILON):
                        priority -= 5.0
                    if(priority > downward_floor_address_dict[downward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_floor = downward_floor

                if(sel_floor > 0):
                    ret_actions[sel_ele] = ElevatorAction(sel_floor, 1)
                    if(downward_floor_address_dict[sel_floor][0] >= 0):
                        ret_actions[downward_floor_address_dict[sel_floor][0]] = ElevatorAction(
                            0, 1)
                        idle_ele_queue.put(
                            downward_floor_address_dict[sel_floor][0])
                    downward_floor_address_dict[sel_floor] = (
                        sel_ele, sel_priority)
                    assigned = True

                # In case no floor is assigned to the current elevator, we
                # search all requiring-upward floor, find the lowest floor and
                # assign it to the elevator
                if(not assigned):
                    if(len(state.RequiringUpwardFloors) > 0):
                        min_unassigned_up_floor = HUGE
                        for upward_floor in state.RequiringUpwardFloors:
                            if(upward_floor_address_dict[upward_floor][0] < 0 and min_unassigned_up_floor > upward_floor):
                                min_unassigned_up_floor = upward_floor
                        if(min_unassigned_up_floor >= 0 and min_unassigned_up_floor < HUGE - 1):
                            ret_actions[sel_ele] = ElevatorAction(
                                min_unassigned_up_floor, 1)
                            priority = state.ElevatorStates[sel_ele].Floor + \
                                EPSILON - min_unassigned_up_floor
                            upward_floor_address_dict[min_unassigned_up_floor] = (
                                sel_ele, priority)

                # print (sel_ele, "going down, sel floor", ret_actions[sel_ele], sel_priority)

            if(state.ElevatorStates[sel_ele].Direction == 0):
                # in case direction == 0,  select the closest requirements
                sel_floor = -1
                sel_priority = -HUGE
                sel_direction = 0

                for upward_floor in state.RequiringUpwardFloors:
                    priority = -abs(upward_floor -
                                    state.ElevatorStates[sel_ele].Floor)
                    if(priority > upward_floor_address_dict[upward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_direction = 1
                        sel_floor = upward_floor

                for downward_floor in state.RequiringDownwardFloors:
                    priority = -abs(downward_floor -
                                    state.ElevatorStates[sel_ele].Floor)
                    if(priority > downward_floor_address_dict[downward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_direction = - 1
                        sel_floor = downward_floor

                if(sel_floor > 0):
                    ret_actions[sel_ele] = ElevatorAction(
                        sel_floor, sel_direction)
                    if(sel_direction > 0):
                        if(upward_floor_address_dict[sel_floor][0] >= 0):
                            idle_ele_queue.put(
                                upward_floor_address_dict[sel_floor][0])
                        upward_floor_address_dict[sel_floor] = (
                            sel_ele, sel_priority)
                    else:
                        if(downward_floor_address_dict[sel_floor][0] >= 0):
                            idle_ele_queue.put(
                                downward_floor_address_dict[sel_floor][0])
                        downward_floor_address_dict[sel_floor] = (
                            sel_ele, sel_priority)
                # print(sel_ele, "stay still, sel floor", ret_actions[sel_ele], sel_priority)

        # print min_unaddressed_up_lift, max_unaddressed_down_lift,
        # state.RequiringUpwardFloors, state.RequiringDownwardFloors
        return ret_actions
