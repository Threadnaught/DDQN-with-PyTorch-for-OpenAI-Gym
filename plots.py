import torch
from torch import nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1

env = gym.make("CartPole-v1")

Q1 = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q2 = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q1.load_state_dict(torch.load('saved/q1.pt'))
Q2.load_state_dict(torch.load('saved/q2.pt'))

# ang_abs = 0.25

angs = np.arange(-0.25, 0.25, 0.25 / 10, dtype=np.float32)
# ang_vels = np.arange(3, -3, 3 / 10, dtype=np.float32)
ang_vels = np.linspace(3,-3, 20, dtype=np.float32)

angs_grid, ang_vels_grid = np.meshgrid(angs, ang_vels)

# print(angs_grid)
# exit()

# print(angs_grid.shape, ang_vels_grid.shape)
# exit()

zeros = np.zeros_like(angs_grid, dtype=np.float32)

states = torch.permute(torch.asarray([zeros, zeros, angs_grid, ang_vels_grid]), [1,2,0])
states_flat = torch.reshape(states, [-1, env.observation_space.shape[0]])

output_q1 = Q1(states_flat).detach()
# output_q2 = Q2(states_flat).detach()

# plt.plot(angs, output_q1[:,0])
# plt.plot(angs, output_q1[:,1])

output_q1_shaped = torch.reshape(output_q1, list(angs_grid.shape[:2]) +[-1])

print(output_q1_shaped.shape)

chosen_action = output_q1_shaped[:,:,0] - output_q1_shaped[:,:,1]




# Nicked from mpl docs
top = cm.get_cmap('Oranges', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(1, 0.33, 128)),
                       bottom(np.linspace(0.33, 1, 128))))
newcmp = colors.ListedColormap(newcolors, name='OrangeBlue')


plt.title('Pole Angle v Angular Velocity')

plt.imshow(chosen_action.detach(), cmap=newcmp, vmax=25, vmin=-25, extent=[-0.25,0.25,-3,3], aspect='auto')
plt.colorbar(label='Q(s,a0) - Q(s,1)')
plt.xlabel('Pole Angle')
plt.ylabel('Pole Angular Velocity')

plt.show()
# print(angs)