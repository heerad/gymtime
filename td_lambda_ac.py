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
h_actor = 4				# hidden layer size for actor
h_critic = 4			# hidden layer size for critic
lr_actor = 1e-3			# learning rate for actor
lr_critic = 1e-3		# learning rate for critic
num_episodes = 200		# number of episodes
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
action_ph = tf.placeholder(dtype=tf.int32, shape=())

# actor network
with tf.variable_scope('actor', reuse=False):
	actor_hidden = tf.layers.dense(state_ph, h_actor, activation = tf.nn.relu)
	actor_logits = tf.layers.dense(actor_hidden, n_actions)
	actor_logits -= tf.reduce_max(actor_logits) # for numerical stability
	actor_policy = tf.nn.softmax(actor_logits)
	actor_logprob_action = actor_logits[0,action_ph] - tf.reduce_logsumexp(actor_logits)

# critic network
with tf.variable_scope('critic', reuse=False):
	critic_hidden = tf.layers.dense(state_ph, h_critic, activation = tf.nn.relu)
	critic_value = tf.layers.dense(critic_hidden, 1)

# gradients
actor_logprob_action_grad = tf.gradients(actor_logprob_action, 
	tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
critic_value_grad = tf.gradients(critic_value,
	tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))

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
	actor_elig_trace = []
	critic_elig_trace = []
	for elem in actor_logprob_action_grad:
		actor_elig_trace.append(np.zeros(elem.shape.as_list()))
	for elem in critic_value_grad:
		critic_elig_trace.append(np.zeros(elem.shape.as_list()))

	# Initial state
	observation = env.reset()
	env.render()

	for t in range(max_steps_ep):

		# compute value of current state, its gradients, and action probabilities
		v_s, v_s_grads, action_probs = sess.run([critic_value, critic_value_grad, actor_policy], 
			feed_dict={state_ph: observation[None]})
		v_s = v_s[0,0]

		# get action
		action = np.random.choice(n_actions, p=action_probs[0,:])

		# compute gradient of current action log probability
		action_logprob_grads = sess.run(actor_logprob_action_grad, 
			feed_dict={state_ph: observation[None], action_ph: action})

		# take step
		observation, reward, done, _info = env.step(action)
		env.render()
		total_reward += reward*(gamma**t)

		# compute value of next state
		if done:
			v_s_prime = 0
		else:
			v_s_prime = sess.run(critic_value, feed_dict={state_ph: observation[None]})
			v_s_prime = v_s_prime[0,0]

		# compute TD error
		delta = reward + gamma*v_s_prime - v_s

		# update eligibility traces
		for i, e in enumerate(actor_elig_trace):
			actor_elig_trace[i] = gamma*lambda_actor*e + action_logprob_grads[i]
		for i, e in enumerate(critic_elig_trace):
			critic_elig_trace[i] = gamma*lambda_critic*e + v_s_grads[i]

		# if t%10==0: print('Actor')
		# update parameters
		for i, param in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')):
			# if t%10==0: print(np.linalg.norm(lr_actor*delta*actor_elig_trace[i]) / np.linalg.norm(sess.run(param)) * 1e3)
			_ = sess.run(tf.assign_add(param, lr_actor*delta*actor_elig_trace[i]))
		# if t%10==0: print('Critic')
		for i, param in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')):
			# if t%10==0: print(np.linalg.norm(lr_actor*delta*critic_elig_trace[i]) / np.linalg.norm(sess.run(param)) * 1e3)
			_ = sess.run(tf.assign_add(param, lr_critic*delta*critic_elig_trace[i]))

		if done: break

	print('Episode %2i, Reward: %7.3f'%(ep,total_reward))


# Finalize and upload results
writefile('info.json', json.dumps(info))
env.close()
gym.upload(outdir)