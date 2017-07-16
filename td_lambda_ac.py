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
lambda_actor = 0.9		# TD(\lambda) parameter (0: 1-step TD, 1: MC) for actor
lambda_critic = 0.9		# TD(\lambda) parameter (0: 1-step TD, 1: MC) for critic
h_actor = 64			# hidden layer size for actor
h_critic = 64			# hidden layer size for critic
lr_actor = 1e-3			# learning rate for actor
lr_critic = 1e-3		# learning rate for critic
lr_decay = 0.99			# learning rate decay (per episode)
l2_reg_actor = 1e-6		# L2 regularization factor for actor
l2_reg_critic = 1e-6	# L2 regularization factor for critic
dropout_actor = 0.		# dropout rate for actor (0 = no dropout)
dropout_critic = 0.		# dropout rate for critic (0 = no dropout)
num_episodes = 500		# number of episodes
max_steps_ep = 1000		# default max number of steps per episode (unless env has a lower hardcoded limit)
clip_norm = 10			# maximum gradient norm for clipping
slow_critic_burnin = 1000		# number of steps where slow critic weights are tied to critic weights
update_slow_critic_every = 20	# number of steps to use slow critic as target before updating it to latest critic

# game parameters
env = gym.make(env_to_use)
state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
n_actions = env.action_space.n 								# Assuming discrete action space

# set seeds to 0
env.seed(0)
np.random.seed(0)

# prepare monitorings
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
	lr_decay = lr_decay,
	l2_reg_actor = l2_reg_actor,
	l2_reg_critic = l2_reg_critic,
	dropout_actor = dropout_actor,
	dropout_critic = dropout_critic,
	num_episodes = num_episodes,
	clip_norm = clip_norm,
	slow_critic_burnin = slow_critic_burnin,
	update_slow_critic_every = update_slow_critic_every
)

#####################################################################################################
## Tensorflow

tf.reset_default_graph()

# placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[1,state_dim]) # Needs to be rank >=2
action_ph = tf.placeholder(dtype=tf.int32, shape=()) # for computing policy gradient for action taken
delta_ph = tf.placeholder(dtype=tf.float32, shape=()) # R + gamma*V(S') - V(S) -- for computing grad steps

# episode counter
episodes = tf.Variable(0.0, trainable=False, name='episodes')
episode_inc_op = episodes.assign_add(1)

# actor network
with tf.variable_scope('actor', reuse=False):
	actor_hidden = tf.layers.dense(state_ph, h_actor, activation = tf.nn.relu)
	actor_hidden_drop = tf.layers.dropout(actor_hidden, rate = dropout_actor, training = True)
	actor_logits = tf.squeeze(tf.layers.dense(actor_hidden_drop, n_actions))
	actor_logits -= tf.reduce_max(actor_logits) # for numerical stability
	actor_policy = tf.nn.softmax(actor_logits)
	actor_logprob_action = actor_logits[action_ph] - tf.reduce_logsumexp(actor_logits)

# need a function so that we can create the same graph (but with different weights) for normal critic, and
# the slowly-changing critic used as a target
def generate_critic_network(s, trainable):
	critic_hidden = tf.layers.dense(s, h_critic, activation = tf.nn.relu, trainable = trainable)
	critic_hidden_drop = tf.layers.dropout(critic_hidden, rate = dropout_critic, training = trainable)
	critic_value = tf.squeeze(tf.layers.dense(critic_hidden_drop, 1, trainable = trainable))
	return critic_value

# critic network
with tf.variable_scope('critic', reuse=False):
	critic_value = generate_critic_network(state_ph, trainable = True)

# slowly-changing critic network (for targets)
with tf.variable_scope('slow_critic', reuse=False):
	critic_value_slow = generate_critic_network(state_ph, trainable = False)

# isolate vars for each network
actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
slow_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_critic')

# update values for slowly-changing critic network to match current critic network
update_slow_critic_ops = []
for i, slow_critic_var in enumerate(slow_critic_vars):
	update_slow_critic_op = slow_critic_var.assign(critic_vars[i])
	update_slow_critic_ops.append(update_slow_critic_op)

update_slow_critic_op = tf.group(*update_slow_critic_ops, name='update_slow_critic')

# gradients
actor_logprob_action_grads = tf.gradients(actor_logprob_action, actor_vars)
critic_value_grads = tf.gradients(critic_value, critic_vars)

# collect stuff for actor and critic updates
ac_update_inputs = dict(
	actor = dict(
		vars = actor_vars,
		grads = actor_logprob_action_grads,
		lambda_ = lambda_actor,
		lr = lr_actor,
		l2_reg = l2_reg_actor
	),
	critic = dict(
		vars = critic_vars,
		grads = critic_value_grads,
		lambda_ = lambda_critic,
		lr = lr_critic,
		l2_reg = l2_reg_critic
	)
)

# gradient step ops with eligibility traces
gradient_step_ops = []
trace_reset_ops = []
grad_step_sanity_checks = []
grad_norms = []
grad_max_vals = []
for network in ac_update_inputs: # actor and critic
	net_update_inputs = ac_update_inputs[network]
	with tf.variable_scope(network+'/traces'):
		for var, grad in zip(net_update_inputs['vars'], net_update_inputs['grads']):

			lr = net_update_inputs['lr'] * lr_decay**episodes
			l2_reg = net_update_inputs['l2_reg']
			if 'bias' in var.name: l2_reg = 0	# don't regularize biases
			lambda_ = net_update_inputs['lambda_']
			
			trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
			# Elig trace update: e <- gamma*lambda*e + grad
			trace_op = trace.assign(gamma*lambda_*trace + tf.clip_by_norm(grad, clip_norm = clip_norm))

			# Gradient step, including for L2 regularization
			grad_step_op = var.assign_add(lr * (delta_ph*trace_op - l2_reg*var))

			grad_step_sanity_checks.append(
				tf.norm(lr *delta_ph * trace) / 
				tf.norm(var - lr * delta_ph * trace))
			grad_norms.append(tf.norm(grad))
			grad_max_vals.append(tf.reduce_max(tf.abs(grad)))
			
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

total_steps = 0
for ep in range(num_episodes):

	print('-------------------------------- START OF EPISODE ----------------------------------')

	# Reset rewards to 0
	total_reward = 0

	# Reset eligibility traces to 0
	_ = sess.run(trace_reset_op)

	# Track TD errors and state values for logging
	ts = []
	vs = []
	deltas = []
	ss = []
	gns = []
	gmvs = []

	# Initial state
	observation = env.reset()
	# env.render()

	for t in range(max_steps_ep):

		# compute value of current state and action probabilities
		state_value, action_probs = sess.run([critic_value, actor_policy], 
			feed_dict={state_ph: observation[None]})

		# get action
		action = np.random.choice(n_actions, p=action_probs)

		# take step
		next_observation, reward, done, _info = env.step(action)
		# env.render()
		total_reward += reward*(gamma**t)

		# update the slow critic's weights to match the latest critic if it's time to do so
		if total_steps%update_slow_critic_every == 0:
			_ = sess.run(update_slow_critic_op)

		# compute value of next state
		if done and not env.env._past_limit(): # only consider next state the end state if it wasn't due to timeout
			next_state_value = 0
		elif total_steps < slow_critic_burnin: # don't use the slowly-changing critic just yet
			next_state_value = sess.run(critic_value, feed_dict={state_ph: next_observation[None]})
		else: # use a slowly-changing critic for the target to improve training stability
			next_state_value = sess.run(critic_value_slow, feed_dict={state_ph: next_observation[None]})

		# compute TD error
		delta = reward + gamma*next_state_value - state_value
		if t%1==0:
			ts.append('%7.3f'%(t)) 
			vs.append('%7.3f'%(state_value))
			deltas.append('%7.3f'%(delta))

		# update actor and critic params using TD(lambda)
		_ = sess.run(td_lambda_op,
			feed_dict={state_ph: observation[None], action_ph: action, delta_ph: delta})

		sanity = sess.run(grad_step_sanity_checks, 
				feed_dict={state_ph: observation[None], action_ph: action, delta_ph: delta})
		gn = sess.run(grad_norms, 
				feed_dict={state_ph: observation[None], action_ph: action, delta_ph: delta})
		gmv = sess.run(grad_max_vals, 
				feed_dict={state_ph: observation[None], action_ph: action, delta_ph: delta})
		if t%1==0:
			ss.append('%7.3f'%(max(np.abs(sanity))))
			gns.append('%7.3f'%(max(gn)))
			gmvs.append('%7.3f'%(max(gmv)))

		observation = next_observation
		total_steps += 1
		if done: 
			_ = sess.run(episode_inc_op)
			break

	print('Episode %2i, Reward: %7.3f'%(ep,total_reward))
	# ts.append('%7.3f'%(t))
	# print(ts)
	vs.append('%7.3f'%(state_value))
	print(vs)
	# deltas.append('%7.3f'%(delta))
	# print(deltas)
	# ss.append('%7.3f'%(max(np.abs(sanity))))
	# print(ss)
	# gns.append('%7.3f'%(max(gn)))
	# print(gns)
	# gmvs.append('%7.3f'%(max(gmv)))
	# print(gmvs)
	print('-------------------------------- END OF EPISODE ----------------------------------')


# Finalize and upload results
writefile('info.json', json.dumps(info))
env.close()
gym.upload(outdir)



# nan rewards => 
	# gradient clipping
	# delayed critic for target (in delta)
	# lr decay
	# l2 reg
# dropout
# shared parameters between actor and critic

	# should there be 0 V(s') award when you finish succesfully?