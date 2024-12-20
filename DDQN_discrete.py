import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
	def __init__(self, len):
		self.rewards = collections.deque(maxlen=len)
		self.state = collections.deque(maxlen=len)
		self.action = collections.deque(maxlen=len)
		self.is_done = collections.deque(maxlen=len)

	def update(self, state, action, reward, done):
		# if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
		# and actions whcih leads to a mismatch when we sample from memory.
		if not done:
			self.state.append(state)
		self.action.append(action)
		self.rewards.append(reward)
		self.is_done.append(done)

	def sample(self, batch_size):
		"""
		sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
		"""
		n = len(self.is_done)
		idx = random.sample(range(0, n-1), batch_size)

		state = np.array(self.state)
		action = np.array(self.action)
		return torch.Tensor(state)[idx].to(device), torch.LongTensor(action)[idx].to(device), \
			   torch.Tensor(state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
			   torch.Tensor(self.is_done)[idx].to(device)

	def reset(self):
		self.rewards.clear()
		self.state.clear()
		self.action.clear()
		self.is_done.clear()


def select_action(model, env, state, eps):
	state = torch.Tensor(state).to(device)
	with torch.no_grad():
		values = model(state)

	# select a random action wih probability eps
	if random.random() <= eps:
		action = np.random.randint(0, env.action_space.n)
	else:
		action = np.argmax(values.cpu().numpy())

	return action


def train(batch_size, current, target, optim, memory, gamma):

	states, actions, next_states, rewards, is_done = memory.sample(batch_size)

	q_values = current(states)

	next_q_values = current(next_states)
	next_q_state_values = target(next_states)

	q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
	next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
	expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

	loss = (q_value - expected_q_value.detach()).pow(2).mean()

	optim.zero_grad()
	loss.backward()
	optim.step()


def evaluate(Qmodel, env, repeats):
	"""
	Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
	episode reward.
	"""
	Qmodel.eval()
	perform = 0
	for _ in range(repeats):
		state = env.reset()
		done = False
		while not done:
			state = torch.Tensor(state).to(device)
			with torch.no_grad():
				values = Qmodel(state)
			action = np.argmax(values.cpu().numpy())
			state, reward, done, _ = env.step(action)
			perform += reward
	Qmodel.train()
	return perform/repeats


def main(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.999, eps_min=0.01, update_step=10, batch_size=64, update_repeats=25,
		 num_episodes=3000, seed=42, max_memory_size=5000, measure_step=100, measure_repeats=100, hidden_dim=64, env_name='CartPole-v1',
		 render=False, render_step=50, q1_var_record_step=1000, double=True):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	env = gym.make(env_name)
	env.seed(seed)

	Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
						hidden_dim=hidden_dim).to(device)
	Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
					hidden_dim=hidden_dim).to(device)

	optimizer_1 = torch.optim.Adam(Q_1.parameters(), lr=lr)
	optimizer_2 = torch.optim.Adam(Q_2.parameters(), lr=lr)

	memory_1 = Memory(max_memory_size)
	memory_2 = Memory(max_memory_size)

	if not double:
		Q_1 = Q_2
		optimizer_1 = optimizer_2
		memory_1 = memory_2

	performance = []
	episode_rewards = []
	epsilons = []
	Q_1_history = {}

	for episode in range(num_episodes):
		# display the performance
		if (episode % measure_step == 0) and episode >= min_episodes:
			performance.append([episode, evaluate(Q_1, env, measure_repeats), evaluate(Q_2, env, measure_repeats)])
			print("Episode: ", episode)
			print("rewards: ", performance[-1][1:])
			print("lr: ", lr)
			print("eps: ", eps)
		
		if episode % 2 == 0:
			Q = Q_1
			memory = memory_1
			q_i = 1
		else:
			Q = Q_2
			memory = memory_2
			q_i = 2

		state = env.reset()
		memory.state.append(state)

		done = False
		i = 0

		if double:
			episode_rewards.append([q_i, 0])
		else:
			episode_rewards.append([1, 0])
		
		epsilons.append(eps)

		while not done:
			i += 1
			action = select_action(Q, env, state, eps)

			state, reward, done, _ = env.step(action)

			# render the environment if render == True
			if render and episode % render_step == 0:
				env.render()

			# save state, action, reward sequence
			memory.update(state, action, reward, done)
			episode_rewards[-1][1] += reward

		if episode >= min_episodes and episode % update_step == 0:
			for _ in range(update_repeats):
				train(batch_size, Q_1, Q_2, optimizer_1, memory_1, gamma)
				train(batch_size, Q_2, Q_1, optimizer_2, memory_2, gamma)
		
		if episode > 0 and episode % q1_var_record_step == 0:
			Q_1_copy = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
									hidden_dim=hidden_dim).to(device)
			Q_1_copy.load_state_dict(Q_1.state_dict())
			Q_1_history[episode] = Q_1_copy

		# update eps
		eps = max(eps*eps_decay, eps_min)

	return Q_1, Q_2, performance, episode_rewards, epsilons, Q_1_history