import sys
import random
from intrabuildingtransport.mansion.utils import EPSILON, HUGE

class DispatcherBase(object):
  '''
  A basic demonstration of dispatcher
  A dispatcher must provide policy and feedback function
  The policy function receives MansionState and output ElevatorAction Lists
  The feedback function receives reward
  '''
  def link_mansion(self, mansion):
    self._mansion = mansion

  def load_settings(self):
    pass

  def feedback(self, state, action, r):
    pass

  def policy(self, state):
    raise NotImplementedError()
