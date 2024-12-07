import torch
from torch import nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from DDQN_discrete import *

env = gym.make("CartPole-v1")
env.reset()

Q1_single = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q1_single.load_state_dict(torch.load('saved-single/q1.pt', map_location=torch.device('cpu')))
Q1_double = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q1_double.load_state_dict(torch.load('saved-double/q1.pt', map_location=torch.device('cpu')))


angs = np.linspace(-0.25, 0.25, 20, dtype=np.float32)
zeros = np.zeros_like(angs)

states = torch.permute(torch.asarray([zeros, zeros, angs, zeros]), [1,0])

Q1_single_output = Q1_single(states)
Q1_single_left_val = Q1_single_output[:,1]
Q1_single_right_val = Q1_single_output[:,0]

Q1_double_output = Q1_double(states)
Q1_double_left_val = Q1_double_output[:,1]
Q1_double_right_val = Q1_double_output[:,0]

plt.plot(angs, Q1_single_left_val.detach().cpu().numpy(), '--', label='Q1 DQN output (left)', color='C0')
plt.plot(angs, Q1_single_right_val.detach().cpu().numpy(), '--', label='Q1 DQN output (right)', color='C1')

plt.plot(angs, Q1_double_left_val.detach().cpu().numpy(), label='Q1 DQDN output (left)', color='C0')
plt.plot(angs, Q1_double_right_val.detach().cpu().numpy(), label='Q1 DQDN output (right)', color='C1')

plt.legend()

plt.show()