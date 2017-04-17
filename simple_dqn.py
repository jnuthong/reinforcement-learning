# -*- encoding: utf-8 -*-
import gym
import numpy as np 
import os
import sys
import tensorflow as tf 
from collections import defaultdict
import random

# from lib import plotting
from collections import deque
from collections import namedtuple

env = gym.envs.make("Breakout-v0")

VALID_ACTIONS = [0, 1, 2, 3]

class StateProcess():
	"""
	Process a raw Atari Screen Image
	"""
	def __init__(self):
		"""
		"""
		with tf.variable_scope("state_process"):
			self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
			self.output = tf.image.rgb_to_grayscale(self.input_state)
			self.output = tf.image.resize_images(self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)

	def process(self, sess, state):
		"""
		"""
		return sess.run(self.output, {self.input_state: state})

class Estimator():
	"""
	Q-value Estimator neural network.
	This network is used for both the Q-Network and the Target Network
	"""

	def __init__(self, scope="default", summaries_dir=None):
		"""
		"""
		self.scope = scope
		with tf.variable_scope(scope):
			self.x = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="x")
			self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
			self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

			X = tf.to_float(self.x) / 255.0
			batch_size = tf.shape(self.x)[0]

			# three convolution layer
			conv1 = tf.contrib.layers.conv2d(X, 32, [8, 8], 4, activation_fn=tf.nn.relu)
			conv2 = tf.contrib.layers.conv2d(conv1, 64, [4, 4], 2, activation_fn=tf.nn.relu)
			conv3 = tf.contrib.layers.conv2d(conv2, 64, [3, 3], 1, activation_fn=tf.nn.relu)

			# fully connected layers
			flattened = tf.contrib.layers.flatten(conv3)
			fc1 = tf.contrib.layers.fully_connected(flattened, 512)
			self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

			# get the prediction for the chosen action
			gather_indices = tf.range(batch_size) * len(VALID_ACTIONS) + self.actions
			self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

			# Calculate loss
			self.losses = tf.squared_difference(self.y, self.action_predictions)
			self.loss = tf.reduce_mean(self.losses)

			# Optimizer Parameters from original paper
			self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.95, 0.01)
			self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

			self.summaries = tf.summary.merge([
					tf.summary.scalar("loss", self.loss),
					tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
				])

			if summaries_dir:
				summaries_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
				if not os.path.exists(summaries_dir):
					os.makedirs(summaries_dir)
				self.summar_writer = tf.summary.FileWriter(summaries_dir)

	def predict(self, sess, s):
		"""
		"""
		return sess.run(self.predictions, { self.x: s})

	def update(self, sess, s, a, y):
		"""
		"""
		feed_dict = {self.x : s, self.y: y, self.actions: a}
		summaries, global_step, _, loss = sess.run(
			[self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss], 
			feed_dict)
		if self.summar_writer:
			self.summar_writer.add_summary(summaries, global_step)
		return loss

def copy_model_parameters(sess, estimator1, estimator2):
	"""
	"""
	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):
		op = e2_v.assign(e1_v)
		update_ops.append(op)

	sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
	"""
	"""
	def policy_fn(sess, observation, epsilon):
		A = np.ones(nA, dtype=float) * epsilon / nA
		q_values = estimator.predict(sess, observation)
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A

	return policy_fn

def deep_q_learning(sess,
					env,
					q_estimator,
					target_estimator,
					state_processor,
					num_episodes,
					experiment_dir,
					replay_memory_size=500000,
					replay_memory_init_size=5000,
					update_target_estimator_every=10000,
					discount_factor=0.99,
					epsilon_start=1.0,
					epsilon_end=0.1,
					epsilon_decay_steps=500000,
					batch_size=32,
					record_video_every=50):
	"""
	"""
	
	_T = 4
	total_t = 0
	relay_memory = deque(maxlen=replay_memory_size)

	episode_rewards = defaultdict(float)
	episode_lengths = defaultdict(float) 

	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	saver = tf.train.Saver()

	for i in range(num_episodes):

		state = env.reset()
	
		policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))
		epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
		_state = np.array([state_processor.process(sess, state)] * 4)
		action_probs = policy(sess, [np.stack(_state, axis=2)], epsilon)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
	
		done = False

		# each 4 frame as a step
		_state = [state_processor.process(sess, state)] 
		_r = 0

		# in origin paper 
		"""
		'..., the agent sees and selects action on every kth frame instead of every frame, 
		and its last action is repeated on skipped frames...'
		"""
		for j in range(_T - 1):
			state, reward, done, _ = env.step(action)
			_r += reward
			_state.append(state_processor.process(sess, state))

		if len(relay_memory) == replay_memory_size:
			relay_memory.popleft()

		while done is False:
			# add 4-frame into D 
			if len(_state) == _T:
				if len(relay_memory) == replay_memory_size:
					relay_memory.popleft()

				episode_rewards[i] += _r
				episode_lengths[i] += 1.0 
				total_t += 1

				relay_memory.append((_state, action, _r, [state_processor.process(sess, state)] * 4))

				# resize input state
				policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))
				epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
				action_probs = policy(sess, [np.stack(_state, axis=2)], epsilon)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

				_state = []
				_r = 0

				# get enough train sample before learning
				if len(relay_memory) % 1000 == 0: print("replay_memory_init_size {}, Memory Length {}".format(replay_memory_init_size, len(relay_memory)))
				if replay_memory_init_size >= len(relay_memory): continue

				samples = random.sample(relay_memory, batch_size)
				states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

				next_states_batch = [np.stack(ele, axis=2) for ele in next_states_batch]
				q_values = target_estimator.predict(sess, next_states_batch)
				target_y = reward_batch + discount_factor * np.amax(q_values, axis=1)

				states_batch = [np.stack(ele, axis=2) for ele in states_batch]
				# print("state_batch {}, action_batch {}, target_y {}".format(len(states_batch), len(action_batch), len(target_y)))
				loss = q_estimator.update(sess, states_batch, action_batch, target_y)

				if total_t % update_target_estimator_every == 0:
					copy_model_parameters(sess, q_estimator, target_estimator)
					print("Copy Model Parameters to Target Network. \n")

					save_path = os.path.abspath("./checkpoint/model.ckpt")
					if not os.path.exists(save_path):
						os.makedirs(save_path)

					_local_save = saver.save(sess, os.path.abspath("./checkpoint/model.ckpt"))

					print("Step {} in Episode - {}, with reward {}, total_step {}".format(episode_lengths[i], i, episode_rewards[i], total_t))
					sys.stdout.flush()
			
			state, reward, done, _ = env.step(action)
			_r += reward
			_state.append(state_processor.process(sess, state))

		# if end before _state have 4 frame, then we fill with last frame
		if len(_state) < _T:
			_local = _state[-1]
			_state += [_local] * (_T - len(_state))
		
			if len(relay_memory) == replay_memory_size:
				relay_memory.popleft()
			
			relay_memory.append((_state, action, _r, [state_processor.process(sess, state)] * 4))
			
			episode_rewards[i] += _r
			episode_lengths[i] += 1.0 
			total_t += 1

		print("Step {} in Episode - {}, with reward {}, total_step {}".format(episode_lengths[i], i, episode_rewards[i], total_t))

		episode_summary = tf.Summary()
		episode_summary.value.add(simple_value=episode_rewards[i], node_name="episode_reward", tag="episode_reward")
		episode_summary.value.add(simple_value=episode_lengths[i], node_name="episode_length", tag="episode_length")
		q_estimator.summar_writer.add_summary(episode_summary, total_t)
		q_estimator.summar_writer.flush()

tf.reset_default_graph()

state_processor = StateProcess()

experiment_dir = os.path.abspath("./experiment/{}".format(env.spec.id))

global_step = tf.Variable(0, name='global_step', trainable=False)

q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	deep_q_learning(sess,
					env, 
					q_estimator=q_estimator,
					target_estimator=target_estimator,
					state_processor=state_processor,
					experiment_dir=experiment_dir,
					num_episodes=10000,
					replay_memory_size=500000,
					replay_memory_init_size=5000,
					update_target_estimator_every=10000,
					epsilon_start=1.0,
					epsilon_end=0.1,
					epsilon_decay_steps=1000000,
					discount_factor=0.99,
					batch_size=32)