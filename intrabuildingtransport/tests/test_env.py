import sys
sys.path.append('.')
from intrabuildingtransport import simulator
from intrabuildingtransport.mansion.utils import ElevatorAction


env = simulator.Simulator("config.ini")
env.seed(1998)
iteration = env.iterations
step = env.reset()
action = [ElevatorAction(-1, 1) for i in range(4)]
for i in range(100):
    next_state, reward, _ = env.step(action)