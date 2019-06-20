import sys
import random
import parl
import numpy as np
from intrabuildingtransport.mansion.utils import ElevatorState, ElevatorAction, MansionState
from intrabuildingtransport.mansion.utils import EPSILON, HUGE
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.mansion_manager import MansionManager

def discretize(value, n_dim, min_val, max_val):
  '''
  discretize a value into a vector of n_dim dimension 1-hot representation
  with the value below min_val being [1, 0, 0, ..., 0]
  and the value above max_val being [0, 0, ..., 0, 1]
  '''
  assert n_dim > 1
  delta = (max_val - min_val) / float(n_dim - 1)
  active_pos = int((value - min_val)/delta + 0.5)
  active_pos = min(n_dim - 1, active_pos)
  active_pos = max(0, active_pos)
  ret_array = [0 for i in range(n_dim)]
  ret_array[active_pos] = 1.0
  return ret_array

def linear_discretize(value, n_dim, min_val, max_val):
  '''
  discretize a value into a vector of n_dim dimensional representation
  with the value below min_val being [1, 0, 0, ..., 0]
  and the value above max_val being [0, 0, ..., 0, 1]
  e.g. if n_dim = 2, min_val = 1.0, max_val = 2.0
    if value  = 1.5 returns [0.5, 0.5], if value = 1.8 returns [0.2, 0.8]
  '''
  assert n_dim > 1
  delta = (max_val - min_val) / float(n_dim - 1)
  active_pos = int((value - min_val)/delta + 0.5)
  active_pos = min(n_dim - 2, active_pos)
  active_pos = max(0, active_pos)
  anchor_pt = active_pos * delta + min_val
  if(anchor_pt > value and anchor_pt > min_val + 0.5 * delta):
    anchor_pt -= delta
    active_pos -= 1
  weight = (value - anchor_pt) / delta
  weight = min(1.0, max(0.0, weight))
  ret_array = [0 for i in range(n_dim)]
  ret_array[active_pos] = 1.0 - weight
  ret_array[active_pos + 1] = weight
  return ret_array

def clip(value, min_val, max_val):
  return min(max_val, max(min_val, value))

def ele_state_preprocessing(ele_state):
  ele_feature = []

  #add floor information
  ele_feature.extend(linear_discretize(ele_state.Floor, ele_state.MaximumFloor, 1.0, ele_state.MaximumFloor))

  #add velocity information
  ele_feature.extend(linear_discretize(ele_state.Velocity, 21, - ele_state.MaximumSpeed,  ele_state.MaximumSpeed))

  #add door information
  ele_feature.append(ele_state.DoorState)
  ele_feature.append(float(ele_state.DoorIsOpening))
  ele_feature.append(float(ele_state.DoorIsClosing))

  #add direction information
  ele_feature.extend(discretize(ele_state.Direction, 3, -1, 1))

  #add load weight information
  ele_feature.extend(linear_discretize(ele_state.LoadWeight / ele_state.MaximumLoad, 5, 0.0, 1.0))

  #add other information
  target_floor_binaries = [0.0 for i in range(ele_state.MaximumFloor)]
  for target_floor in ele_state.ReservedTargetFloors:
    target_floor_binaries[target_floor - 1] = 1.0
  ele_feature.extend(target_floor_binaries)

  dispatch_floor_binaries = [0.0 for i in range(ele_state.MaximumFloor + 1)]
  dispatch_floor_binaries[ele_state.CurrentDispatchTarget] = 1.0
  ele_feature.extend(dispatch_floor_binaries)
  ele_feature.append(ele_state.DispatchTargetDirection)

  return ele_feature

def obs_dim(mansion):
  assert isinstance(mansion, MansionManager)
  ele_dim = mansion._floor_number * 3 + 34 
  obs_dim = ele_dim * mansion._elevator_number + mansion._floor_number * 2 + mansion._elevator_number
  return obs_dim

def act_dim(mansion):
  assert isinstance(mansion, MansionManager)
  return mansion._floor_number * 2 + 2

def mansion_state_preprocessing(mansion_state):
  ele_features = list()
  for ele_state in mansion_state.ElevatorStates:
    ele_features.append(ele_state_preprocessing(ele_state))
    max_floor = ele_state.MaximumFloor

  target_floor_binaries_up = [0.0 for i in range(max_floor)]
  target_floor_binaries_down = [0.0 for i in range(max_floor)]
  for floor in mansion_state.RequiringUpwardFloors:
    target_floor_binaries_up[floor - 1] = 1.0
  for floor in mansion_state.RequiringDownwardFloors:
    target_floor_binaries_down[floor - 1] = 1.0
  target_floor_binaries = target_floor_binaries_up + target_floor_binaries_down

  idx = 0
  man_features = list()
  for idx in range(len(mansion_state.ElevatorStates)):
    elevator_id_vec = discretize(idx + 1, len(mansion_state.ElevatorStates), 1, len(mansion_state.ElevatorStates))
    idx_array = list(range(len(mansion_state.ElevatorStates)))
    idx_array.remove(idx)
    random.shuffle(idx_array)
    man_features.append(ele_features[idx])
    for left_idx in idx_array:
      man_features[idx] = man_features[idx] + ele_features[left_idx]
    man_features[idx] = man_features[idx] + elevator_id_vec + target_floor_binaries
  return np.asarray(man_features, dtype='float32')

def action_idx_to_action(action_idx, act_dim):
  assert isinstance(action_idx, int)
  assert isinstance(act_dim, int)
  realdim = act_dim - 2
  if(action_idx == realdim):
    return ElevatorAction(0, 1)
  elif(action_idx == realdim + 1):
    return ElevatorAction(-1, 1)
  action = action_idx
  if(action_idx < realdim / 2):
    direction = 1
    action += 1
  else:
    direction = -1
    action -= realdim / 2
    action += 1
  return ElevatorAction(action, direction)

def action_to_action_idx(action, act_dim):
  assert isinstance(action, ElevatorAction)
  assert isinstance(act_dim, int)
  realdim = act_dim - 2
  if(action.TargetFloor == 0):
    return realdim
  elif(action.TargetFloor < 0):
    return realdim + 1
  action_idx = 0
  if(action.DirectionIndicator < 0):
    action_idx += int(realdim / 2)
  action_idx += action.TargetFloor - 1
  return action_idx
