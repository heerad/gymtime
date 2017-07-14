import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
import json, sys, os
from os import path

#####################################################################################################
## Setup

env_to_use = 'CartPole-v0'

# hyperparameters
gamma = 1.				# reward discount factor
lambda_actor = 0.8		# TD(\lambda) parameter (0: 1-step TD, 1: MC) for actor
lambda_critic = 0.8		# TD(\lambda) parameter (0: 1-step TD, 1: MC) for critic
h_actor = 128			# hidden layer size for actor
h_critic = 128			# hidden layer size for critic
lr_actor = 1e-3			# learning rate for actor
lr_critic = 1e-3		# learning rate for critic
num_episodes = 500		# number of episodes
max_steps_ep = 200		# maximum number of timesteps to wait for `done` per episode

# game parameters
env = gym.make(env_to_use)
state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
n_actions = env.action_space.n 								# Assuming discrete action space

# set seeds to 0
env.seed(0)
np.random.seed(0)

# prepare monitoring
outdir = '/tmp/td_lambda_ac-agent-results'
env = wrappers.Monitor(env, outdir, force=True)
def writefile(fname, s):
    with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
info = {}
info['env_id'] = env.spec.id
info['params'] = dict(
	gamma = gamma,
	lambda_actor = lambda_actor,
	lambda_critic = lambda_critic,
	h_actor = h_actor,
	h_critic = h_critic,
	lr_actor = lr_actor,
	lr_critic = lr_critic,
	num_episodes = num_episodes,
	max_steps_ep = max_steps_ep
)

#####################################################################################################
## Tensorflow

tf.reset_default_graph()

# placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[1,state_dim]) # Needs to be rank >=2
action_ph = tf.placeholder(dtype=tf.int32, shape=()) # for computing policy gradient for action taken
delta_ph = tf.placeholder(dtype=tf.float32, shape=()) # R + gamma*V(S') - V(S) -- for computing grad steps

# actor network
with tf.variable_scope('actor', reuse=False):
	actor_hidden = tf.layers.dense(state_ph, h_actor, activation = tf.nn.relu)
	actor_logits = tf.squeeze(tf.layers.dense(actor_hidden, n_actions))
	actor_logits -= tf.reduce_max(actor_logits) # for numerical stability
	actor_policy = tf.nn.softmax(actor_logits)
	actor_logprob_action = actor_logits[action_ph] - tf.reduce_logsumexp(actor_logits)

# critic network
with tf.variable_scope('critic', reuse=False):
	critic_hidden = tf.layers.dense(state_ph, h_critic, activation = tf.nn.relu)
	critic_value = tf.squeeze(tf.layers.dense(critic_hidden, 1))

# isolate vars for each network
actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

# gradients
actor_logprob_action_grads = tf.gradients(actor_logprob_action, actor_vars)
critic_value_grads = tf.gradients(critic_value, critic_vars)

# collect stuff for actor and critic updates
ac_update_inputs = dict(
	actor = dict(
		vars = actor_vars,
		grads = actor_logprob_action_grads,
		lambda_ = lambda_actor,
		lr = lr_actor
	),
	critic = dict(
		vars = critic_vars,
		grads = critic_value_grads,
		lambda_ = lambda_critic,
		lr = lr_critic
	)
)

# gradient step ops with eligibility traces
gradient_step_ops = []
trace_reset_ops = []
for network in ac_update_inputs: # actor and critic
	net_update_inputs = ac_update_inputs[network]
	with tf.variable_scope(network+'/traces'):
		for var, grad in zip(net_update_inputs['vars'], net_update_inputs['grads']):
			print('var shape: {}'.format(var.get_shape()))
			print('grad shape: {}'.format(grad.get_shape()))
			trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
			# Elig trace update: e <- gamma*lambda*e + grad
			trace_op = trace.assign(gamma * net_update_inputs['lambda_'] * trace + grad)
			grad_step_op = var.assign_add(net_update_inputs['lr'] * delta_ph * trace_op)
			trace_reset_op = trace.assign(tf.zeros(trace.get_shape()))
			gradient_step_ops.append(grad_step_op)
			trace_reset_ops.append(trace_reset_op)

td_lambda_op = tf.group(*gradient_step_ops, name='td_lambda')
trace_reset_op = tf.group(*trace_reset_ops, name='trace_reset')

# initialize session
sess = tf.Session()	
sess.run(tf.global_variables_initializer())

#####################################################################################################
## Training

# num episodes, max steps per episode
# per ep: running reward, eligibility traces, state
# render
for ep in range(num_episodes):

	# Reset rewards to 0
	total_reward = 0

	# Reset eligibility traces to 0
	_ = sess.run(trace_reset_op)

	# Initial state
	observation = env.reset()
	env.render()

	for t in range(max_steps_ep):

		# compute value of current state and action probabilities
		v_s, action_probs = sess.run([critic_value, actor_policy], 
			feed_dict={state_ph: observation[None]})

		# get action
		action = np.random.choice(n_actions, p=action_probs)

		# take step
		next_observation, reward, done, _info = env.step(action)
		env.render()
		total_reward += reward*(gamma**t)

		# compute value of next state
		if done:
			v_s_prime = 0
		else:
			v_s_prime = sess.run(critic_value, feed_dict={state_ph: next_observation[None]})

		# compute TD error
		delta = reward + gamma*v_s_prime - v_s

		# update actor and critic params using TD(lambda)
		_ = sess.run(td_lambda_op,
			feed_dict={state_ph: observation[None], action_ph: action, delta_ph: delta})

		observation = next_observation
		if done: break

	print('Episode %2i, Reward: %7.3f'%(ep,total_reward))


# Finalize and upload results
writefile('info.json', json.dumps(info))
env.close()
gym.upload(outdir)