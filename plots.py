import torch
from torch import nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from DDQN_discrete import *

# Set up env
env = gym.make("CartPole-v1")
env.reset()



# Set up network
Q1_single = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q1 = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q2 = QNetwork(env.action_space.n, env.observation_space.shape[0], 64)
Q1.load_state_dict(torch.load('saved-double/q1.pt', map_location=torch.device('cpu')))
Q2.load_state_dict(torch.load('saved-double/q2.pt', map_location=torch.device('cpu')))
Q1_single.load_state_dict(torch.load('saved-single/q1.pt', map_location=torch.device('cpu')))

rewards = np.load('saved-double/rewards.npy')
performance = np.load('saved-double/performance.npy')
epsilons = np.load('saved-double/epsilons.npy')

def show_reset_cartpole():
    env.render()

def show_ang_ang_vel_img():
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

def show_episode_reward_graph():
    q_is = np.unique(episode_rewards[:,0])
    i_qs = {int(q_is[i]):i for i in range(len(q_is))}

    # print(x)

    q_ep_rewards = [[] for _ in range(len(q_is))]

    for episode in range(episode_rewards.shape[0]):
        q_ep_rewards[i_qs[episode_rewards[episode,0]]].append([episode, episode_rewards[episode,1]])

    colors_dots = ['C0', 'C1']
    colors_lines = ['C2', 'C3']
    for i in range(len(q_ep_rewards)):
        xs, ys = np.transpose(q_ep_rewards[i])
        plt.plot(xs, ys, 'o', color=colors_dots[i], markeredgewidth=0, alpha=0.25)

        ys_smooth = np.convolve(ys, np.ones(100)/100, mode='valid')
        plt.plot(xs[:len(ys_smooth)]+99, ys_smooth, color=colors_lines[i])

    
    plt.show()
    
    # print(q_ep_rewards[0][:10])
    
    # ep_rewards_split = np.reshape(episode_rewards, [-1, 2])
    # q_1_rewards = ep_rewards_split[:,0]
    # q_2_rewards = ep_rewards_split[:,1]

    # plt.plot(np.arange(0,len(episode_rewards),2), q_1_rewards, 'o', color='#FFA50088', markeredgewidth=0)
    # plt.plot(np.arange(1,len(episode_rewards),2), q_2_rewards, 'o', color='#00FFA588', markeredgewidth=0)
    
    # ep_rewards_moving_average = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
    # plt.plot(np.arange(len(ep_rewards_moving_average))+99, ep_rewards_moving_average)
    # plt.show()

def reward_peformance_graph(chosen_q):
    q_is = np.unique(rewards[:,0])
    i_qs = {int(q_is[i]):i for i in range(len(q_is))}

    q_ep_rewards = [[] for _ in range(len(q_is))]

    for episode in range(rewards.shape[0]):
        q_ep_rewards[i_qs[rewards[episode,0]]].append([episode, rewards[episode,1]])
    
    xs, ys = np.transpose(q_ep_rewards[chosen_q])
    plt.plot(xs, ys, 'o', markeredgewidth=0, alpha=0.5, label='Individual Episode Rewards')

    perf_xs = performance[:,0]
    perf_ys = performance[:,chosen_q+1]

    plt.plot(perf_xs, perf_ys, linewidth=2, label='Averaged Evaluation Rewards')
    plt.legend(loc='lower right')

    plt.ylim(-100,600)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

def performance_comparison_graph():
    fig, ax_rewards = plt.subplots()
    

    perf_xs = performance[:,0]
    perf_q1_ys = performance[:,1]
    perf_q2_ys = performance[:,2]

    plt.plot(perf_xs, perf_q2_ys, linewidth=2, color='C2', label='Q2 Averaged Evaluation Rewards')
    plt.plot(perf_xs, perf_q1_ys, linewidth=2, color='C3', label='Q1 Averaged Evaluation Rewards')

    ax_rewards.set_ylim(-100,600)

    ep_ax = plt.twinx()
    ep_ax.plot(np.arange(epsilons.shape[0]), epsilons, linewidth=2, color='C4', label='epsilon')

    ax_rewards.legend(loc='lower left')
    ep_ax.legend(loc='lower right')
    ep_ax.set_ylim(-1/5,6/5)
    
    ax_rewards.set_xlabel('Episode')
    ax_rewards.set_ylabel('Reward')
    ep_ax.set_ylabel('Epsilon')
    # plt.show()

def plot_q_overestimation(pos):
    plt.figure(figsize=[3.2,4.8])
    angs = np.linspace(-0.25, 0.25, 20, dtype=np.float32)
    zeros = np.zeros_like(angs)
    pos = zeros + pos

    states = torch.permute(torch.asarray([pos, zeros, angs, zeros]), [1,0])

    Q1_single_output = Q1_single(states)
    Q1_single_left_val = Q1_single_output[:,1]
    Q1_single_right_val = Q1_single_output[:,0]

    Q1_double_output = Q1(states)
    Q1_double_left_val = Q1_double_output[:,1]
    Q1_double_right_val = Q1_double_output[:,0]

    print(np.max(Q1_single_output.detach().cpu().numpy()))

    plt.plot(angs, Q1_single_left_val.detach().cpu().numpy(), '--', label='DQN left', color='C0')
    plt.plot(angs, Q1_single_right_val.detach().cpu().numpy(), '--', label='DQN right', color='C1')

    plt.plot(angs, Q1_double_left_val.detach().cpu().numpy(), label='DQDN left', color='C0')
    plt.plot(angs, Q1_double_right_val.detach().cpu().numpy(), label='DQDN right', color='C1')

    plt.ylabel('Q(s,a)')
    plt.xlabel('angle (rads)')

    plt.legend()
    plt.ylim(-150,135)


# reward_peformance_graph(0)
# plt.savefig('images/r-p-graph-q1.png')
# plt.clf()

# reward_peformance_graph(1)
# plt.savefig('images/r-p-graph-q2.png')
# plt.clf()

# performance_comparison_graph()
# plt.savefig('images/p-graph-both-qs.png')
# plt.show()

for pos in [-1,0,1]:
    plt.clf()
    plot_q_overestimation(pos)
    # plt.savefig('images/q-overestimation-%i.png' % pos)
    # plt.show()