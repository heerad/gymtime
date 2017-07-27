import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
import json, sys, os
from os import path
import random
from collections import deque

#####################################################################################################
## Algorithm

# Deep Q-Networks (DQN)
# An off-policy action-value function based approach (Q-learning) that uses either UCB or epsilon-greedy 
# exploration to generate experiences (s, a, r, s'). It uses minibatches of these experiences from replay 
# memory to update the Q-network's parameters, sampled using prioritized experience replay.
# Neural networks are used for function approximation.
# A slowly-changing "target" Q network, as well as gradient norm clipping, are used to improve
# stability and encourage convergence.
# Parameter updates are made via Adam.
# Assumes discrete action spaces!

#####################################################################################################
## Setup

env_to_use = 'MountainCar-v0'

# hyperparameters
gamma = 0.99				# reward discount factor
h1 = 128					# hidden layer 1 size
h2 = 128					# hidden layer 2 size
h3 = 128					# hidden layer 3 size
lr = 1e-3				# learning rate
lr_decay = 1			# learning rate decay (per episode)
l2_reg = 1e-6				# L2 regularization factor
dropout = 0				# dropout rate (0 = no dropout)
num_episodes = 15000		# number of episodes
max_steps_ep = 10000	# default max number of steps per episode (unless env has a lower hardcoded limit)
update_slow_target_every = 100	# number of steps to use slow target as target before updating it to latest weights
train_every = 1			# number of steps to run the policy (and collect experience) before updating network weights
replay_memory_capacity = int(1e5)	# capacity of experience replay memory
priority_alpha = 2.0	# exponent by which to transform TD errors in computing experience priority in replay
priority_beta0 = 0.4	# initial exponent on importance sampling weight for prioritized gradient updates
priority_beta_decay_length = 10000	# length of time over which to linearly anneal priority_beta to 1
minibatch_size = 1024	# size of minibatch from experience replay memory for updates
epsilon_start = 1.0		# probability of random action at start
epsilon_end = 0.05		# minimum probability of random action after linear decay period
epsilon_decay_length = 1e5		# number of steps over which to linearly decay epsilon
epsilon_decay_exp = 0.97		# exponential decay rate after reaching epsilon_end (per episode)
use_ucb_exploration = True 		# flag to chooose between epsilon-greedy and UCB exploration strategies
state_dim_discretization = 10 	# number of buckets per dimension to discretize the state space into for counting num visits
q_function_range = 200		# size of range in which true Q values for optimal strategy lie, used for UCB computation

# game parameters
env = gym.make(env_to_use)
state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
n_actions = env.action_space.n 								# Assuming discrete action space

# set seeds to 0
env.seed(0)
np.random.seed(0)

# prepare monitorings
outdir = '/tmp/dqn-agent-results'
env = wrappers.Monitor(env, outdir, force=True)
def writefile(fname, s):
    with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
info = {}
info['env_id'] = env.spec.id
info['params'] = dict(
	gamma = gamma,
	h1 = h1,
	h2 = h2,
	h3 = h3,
	lr = lr,
	lr_decay = lr_decay,
	l2_reg = l2_reg,
	dropout = dropout,
	num_episodes = num_episodes,
	max_steps_ep = max_steps_ep,
	update_slow_target_every = update_slow_target_every,
	train_every = train_every,
	replay_memory_capacity = replay_memory_capacity,
	priority_alpha = priority_alpha,
	priority_beta0 = priority_beta0,
	priority_beta_decay_length = priority_beta_decay_length,
	minibatch_size = minibatch_size,
	epsilon_start = epsilon_start,
	epsilon_end = epsilon_end,
	epsilon_decay_length = epsilon_decay_length,
	epsilon_decay_exp = epsilon_decay_exp,
	use_ucb_exploration = use_ucb_exploration,
	state_dim_discretization = state_dim_discretization,
	q_function_range = q_function_range
)

np.set_printoptions(threshold=np.nan)

# replay memory setup (should really be implemented as a SumTree for O(logn) instead of O(n) ops)
replay_memory = deque(maxlen=replay_memory_capacity)			# used for O(1) popleft() operation
replay_priorities_exp = deque(maxlen=replay_memory_capacity)	# used to compute sampling probability
priority_beta_step = (1. - priority_beta0) / priority_beta_decay_length # used to increase priority_beta

# adds a transition into replay memory with maximal priority
def add_to_memory(experience):
	if len(replay_memory) >= replay_memory_capacity:
		_ = replay_memory.popleft()
		_ = replay_priorities_exp.popleft()
	replay_memory.append(experience)
	max_priority_exp = 1. if len(replay_priorities_exp) == 0 else max(replay_priorities_exp)
	replay_priorities_exp.append(max_priority_exp)

def sample_from_memory(minibatch_size, priority_beta):
	replay_probs = np.zeros(len(replay_priorities_exp))
	replay_priority_exp_sum = sum(replay_priorities_exp)
	for i, rpe in enumerate(replay_priorities_exp):
		replay_probs[i] = rpe/replay_priority_exp_sum
	minibatch_indices = np.random.choice(len(replay_memory), minibatch_size, p=replay_probs)
	minibatch = []
	imp_samp_weights = []
	for mb_i in minibatch_indices:
		minibatch.append(replay_memory[mb_i])
		imp_samp_weights.append((len(replay_memory)*replay_probs[mb_i])**(-priority_beta))
	imp_samp_weights /= max(imp_samp_weights)
	return minibatch, imp_samp_weights, minibatch_indices

def update_memory_priorities(minibatch_indices, new_priorities):
	for i, mb_i in enumerate(minibatch_indices):
		priority_exp = new_priorities[i]**priority_alpha
		replay_priorities_exp[mb_i] = priority_exp


# exploration setup
if use_ucb_exploration:
	# state-action visited counter (used for a UCB-based exploration strategy via Hoeffding's inequality)
	# see: https://en.wikipedia.org/wiki/Hoeffding%27s_inequality#General_case
	# we choose our confidence level p = t^-4 where t is time steps
	# shape of discrized table will be (number of buckets per state dim * num state dims, number of actions)
	# note that this is a hack since in reality Q(s,a) is estimated via a function approximator, rather than
	# computing separate empirical means for each entry in a table, for which this inequality is valid.
	visited_counter = np.zeros((state_dim_discretization**state_dim,n_actions))
	disc_interval_sizes = (env.observation_space.high - env.observation_space.low) / state_dim_discretization
else:
	epsilon = epsilon_start
	epsilon_linear_step = (epsilon_start-epsilon_end)/epsilon_decay_length

#####################################################################################################
## Tensorflow

tf.reset_default_graph()

# placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim]) # input to Q network
action_ph = tf.placeholder(dtype=tf.int32, shape=[None]) # action indices (indices of Q network output)
reward_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # rewards (go into target computation)
next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim]) # input to slow target network
is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
imp_samp_weights_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # importance sampling weights for each minibatch element
is_training_ph = tf.placeholder(dtype=tf.bool, shape=()) # for dropout

# episode counter
episodes = tf.Variable(0.0, trainable=False, name='episodes')
episode_inc_op = episodes.assign_add(1)

# will use this to initialize both Q network and slowly-changing target network with same structure
def generate_network(s, trainable, reuse):
	hidden = tf.layers.dense(s, h1, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
	hidden_drop = tf.layers.dropout(hidden, rate = dropout, training = trainable & is_training_ph)
	hidden_2 = tf.layers.dense(hidden_drop, h2, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
	hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout, training = trainable & is_training_ph)
	hidden_3 = tf.layers.dense(hidden_drop_2, h3, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
	hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout, training = trainable & is_training_ph)
	action_values = tf.squeeze(tf.layers.dense(hidden_drop_3, n_actions, trainable = trainable, name = 'dense_3', reuse = reuse))
	return action_values

with tf.variable_scope('q_network') as scope:
	# Q network applied to state_ph
	q_action_values = generate_network(state_ph, trainable = True, reuse = False)
	# Q network applied to next_state_ph (for double Q learning)
	q_action_values_next = tf.stop_gradient(generate_network(next_state_ph, trainable = False, reuse = True))

# slow target network
with tf.variable_scope('slow_target_network', reuse=False):
	# use stop_gradient to treat the output values as constant targets when doing backprop
	slow_target_action_values = tf.stop_gradient(generate_network(next_state_ph, trainable = False, reuse = False))

# isolate vars for each network
q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
slow_target_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_network')

# update values for slowly-changing target network to match current critic network
update_slow_target_ops = []
for i, slow_target_var in enumerate(slow_target_network_vars):
	update_slow_target_op = slow_target_var.assign(q_network_vars[i])
	update_slow_target_ops.append(update_slow_target_op)

update_slow_target_op = tf.group(*update_slow_target_ops, name='update_slow_target')

# Q-learning targets y_i for (s,a) from experience replay
# = r_i + gamma*Q_slow(s',argmax_{a}Q(s',a)) if s' is not terminal
# = r_i if s' terminal
# Note that we're using Q_slow(s',argmax_{a}Q(s',a)) instead of max_{a}Q_slow(s',a) to address the maximization bias problem via Double Q-Learning
targets = reward_ph + is_not_terminal_ph * gamma * \
	tf.gather_nd(slow_target_action_values, tf.stack((tf.range(minibatch_size), tf.cast(tf.argmax(q_action_values_next, axis=1), tf.int32)), axis=1))

# Estimated Q values for (s,a) from experience replay
estim_taken_action_vales = tf.gather_nd(q_action_values, tf.stack((tf.range(minibatch_size), action_ph), axis=1))

# 1-step temporal difference errors
td_errors = targets - estim_taken_action_vales

# for prioritized experience replay
abs_td_errors = tf.abs(td_errors)

# loss function (with regularization)
loss = tf.reduce_mean(tf.square(td_errors)*imp_samp_weights_ph)
for var in q_network_vars:
	if not 'bias' in var.name:
		loss += l2_reg * 0.5 * tf.nn.l2_loss(var)

# optimizer
train_op = tf.train.AdamOptimizer(lr*lr_decay**episodes).minimize(loss)

# initialize session
sess = tf.Session()	
sess.run(tf.global_variables_initializer())

#####################################################################################################
## Training

total_steps = 0
for ep in range(num_episodes):

	total_reward = 0
	steps_in_ep = 0

	max_ucb_ep = -1

	# Initial state
	observation = env.reset()
	if ep%10 == 0: env.render()

	for t in range(max_steps_ep):

		# choose between UCB and epsilon greedy for action selection
		if use_ucb_exploration:
			# choose action according to UCB with Hoeffding's inequality applied to the estimated Q values
			q_s = sess.run(q_action_values, 
					feed_dict = {state_ph: observation[None], is_training_ph: False})
			obs_discrete = np.minimum(((observation - env.observation_space.low)/disc_interval_sizes).astype(int), state_dim_discretization-1)
			obs_idx = sum([i_el[1]*state_dim_discretization**i_el[0] for i_el in enumerate(obs_discrete)])
			ucbs = q_function_range * np.sqrt(2*np.log(total_steps)/visited_counter[obs_idx,:])
			action = np.argmax(q_s + ucbs)
			max_ucb_ep = max(max_ucb_ep, max(ucbs))
		else:
			# choose action according to epsilon-greedy policy wrt Q
			if np.random.random() < epsilon:
				action = np.random.randint(n_actions)
			else:
				q_s = sess.run(q_action_values, 
					feed_dict = {state_ph: observation[None], is_training_ph: False})
				action = np.argmax(q_s)

		# take step
		next_observation, reward, done, _info = env.step(action)
		if ep%10 == 0: env.render()
		total_reward += reward

		add_to_memory((observation, action, reward, next_observation, 
			# is next_observation a terminal state?
			# 0.0 if done and not env.env._past_limit() else 1.0))
			0.0 if done else 1.0))

		# update the slow target's weights to match the latest q network if it's time to do so
		if total_steps%update_slow_target_every == 0:
			_ = sess.run(update_slow_target_op)

		# update network weights to fit a minibatch of experience
		if total_steps%train_every == 0 and len(replay_memory) >= minibatch_size:

			# hyperparameter that controls strength of importance-sampling adjustment
			priority_beta = min(priority_beta0 + priority_beta_step*ep, 1.)

			# grab N (s,a,r,s') tuples from replay memory
			# also get importance sampling weights and minibatch indices to deal with prioritized sampling
			minibatch, imp_samp_weights, minibatch_indices = sample_from_memory(minibatch_size, priority_beta)

			# get the absolute TD errors for the minibatch based on the latest model parameters
			# then do a train_op to fit the Q network (using double Q-learning) to the importance-sampled minibatch
			new_priorities, _ = sess.run([abs_td_errors, train_op], 
				feed_dict = {
					state_ph: np.asarray([elem[0] for elem in minibatch]),
					action_ph: np.asarray([elem[1] for elem in minibatch]),
					reward_ph: np.asarray([elem[2] for elem in minibatch]),
					next_state_ph: np.asarray([elem[3] for elem in minibatch]),
					is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
					imp_samp_weights_ph: np.asarray(imp_samp_weights),
					is_training_ph: True})

			# update the minibatch's priorities with the absolute TD errors
			update_memory_priorities(minibatch_indices, new_priorities)

		observation = next_observation
		total_steps += 1
		steps_in_ep += 1

		# update exploration parameters
		if use_ucb_exploration:
			# increment state-action visited counter
			visited_counter[obs_idx,action] = visited_counter[obs_idx,action] + 1
		else:
			old_epsilon = epsilon
			# linearly decay epsilon from epsilon_start to epsilon_end over epsilon_decay_length steps
			if total_steps < epsilon_decay_length:
				epsilon -= epsilon_linear_step
			# then exponentially decay it every episode
			elif done:
				epsilon *= epsilon_decay_exp
		
		if done: 
			# Increment episode counter
			_ = sess.run(episode_inc_op)
			break

	if use_ucb_exploration:
		print('Episode %2i, Reward: %7.3f, Steps: %i, Max UCB: %7.3f'%(ep,total_reward,steps_in_ep, max_ucb_ep))
		# if ep%100 == 0: print(visited_counter)
	else:
		print('Episode %2i, Reward: %7.3f, Steps: %i, Epsilon: %7.3f'%(ep,total_reward,steps_in_ep, old_epsilon))

# Finalize and upload results
writefile('info.json', json.dumps(info))
env.close()
gym.upload(outdir)