import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Add, Lambda, Input
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, datetime, time
import models
from cusotm_atari_preprocessing import AtariPreprocessing
import gym.envs.atari.atari_env
from replay_memory import Storage
from helpers import measure

# utility stuff
GPU = True
if not GPU: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

"""
TODO DEBUG CHECKLIST
# Game State output             |x|         # other outputs not thoroughly tested but looking at them they look right
# Replay Buffer                 |x|         # all methods work as intended and are even acceptably fast
# Actual Network                | |         # to be honest i don't know what could be wrong thee will still try
# step / episode loop           | |         # 
# train_step method             | |         #
# get_action method             | |         #
TWEAK HYPER PARAMS              | |         # at this point this is the most obv one and I dread it

"""

print(', executing_eagerly in print: {}'.format(tf.executing_eagerly()))
tf.print(', executing_eagerly in tf.print: {}'.format(tf.executing_eagerly()))

# Agent parameters
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.1
FRAMES_TO_REACH_FINAL = 1_000_000
EXPONENTIAL_EXPLORATION_DECAY = True  # default is linear
MINIBATCH_SIZE = 32
MAX_REPLAY_MEMORY_SIZE = 200_00
START_TRAINING_AFTER = 20_000
FRAMES_TO_INCLUDE = 4
UPDATE_TARGET_EVERY = 10_000
DISCOUNT = 0.99
ACTION_REPEAT = 4  # ie. agent sees every n th frame
OPTIMIZER_STEP_EVERY = 4  # optimizer is applied every n th action choice
DUELING = True
DOUBLE = True

# Optimizer parameters
LEARN_RATE = 0.00015
BETA_1 = 0.95
BETA_2 = 0.95
EPSILON = 0.01

# Environment parameters
MAX_NO_ACTION = 30
GAME = 'breakout'

# utility objects
env = gym.envs.atari.atari_env.AtariEnv(game=GAME, obs_type='image', frameskip=1)
optimizer = K.optimizers.Adam(LEARN_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)


class Agent:
    def __init__(self, load_from_file=False):
        # params for training
        self.minibatch_size = MINIBATCH_SIZE
        self.max_replay = MAX_REPLAY_MEMORY_SIZE
        self.train_start = START_TRAINING_AFTER
        self.remember_frames = FRAMES_TO_INCLUDE
        self.target_update = UPDATE_TARGET_EVERY
        self.discount = DISCOUNT
        self.action_repeat = ACTION_REPEAT
        self.optimizer_frequency = OPTIMIZER_STEP_EVERY
        self.learn_rate = LEARN_RATE

        if EXPONENTIAL_EXPLORATION_DECAY:
            self.exploration_factor = MIN_EXPLORATION_RATE ** (1 / FRAMES_TO_REACH_FINAL)
        else:
            self.exploration_slope = (MIN_EXPLORATION_RATE - EXPLORATION_RATE) / FRAMES_TO_REACH_FINAL

        self.exploration_decay = False
        self.exploration_rate = EXPLORATION_RATE
        self.min_exploration_rate = MIN_EXPLORATION_RATE
        self.exploration_frames_to_final = FRAMES_TO_REACH_FINAL

        self.env = AtariPreprocessing(env, noop_max=MAX_NO_ACTION)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n

        # models and other objects
        self.step_model = self.get_model(DUELING)
        self.target_model = self.get_model(DUELING)
        self.update_target_model()

        self.replay_memory = Storage(self.max_replay, self.state_shape, self.minibatch_size)
        self.ckpt = tf.train.Checkpoint(model=self.step_model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, './checkpoint', max_to_keep=10)

        # metrics
        self.model_loss = tf.keras.metrics.Mean('model_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.graph_log_dir = f'logs/graph/{current_time}/'
        self.stats_log_dir = f'logs/stats/{current_time}/'
        self.graph_summary_writer = tf.summary.create_file_writer(self.graph_log_dir)
        self.stats_summary_writer = tf.summary.create_file_writer(self.stats_log_dir)

        self.total_steps = 0
        self.total_eps = 0
        self.total_train_steps = 0
        self.steps_per_ep = []
        self.ep_losses = []
        self.ep_rewards = []

        self.step_model.summary()
        self.target_model.summary()

        if load_from_file:
            if DOUBLE or DUELING: names = (f'agent_state_{GAME}.state', f'step_model_{GAME}.h5', f'target_model_{GAME}.h5')
            else: names = (f'agent_state_simple_{GAME}.state', f'step_model_simple_{GAME}.h5', f'target_model_simple_{GAME}.h5')
            with open(names[0], 'rb') as f:
                params = pickle.load(f)
                self.exploration_decay = params[0]
                self.exploration_rate = params[1]
                self.total_steps = params[2]
                self.total_eps = params[3]
                self.total_train_steps = params[4]
                self.replay_memory = params[5]

            self.step_model.load_weights(names[1])
            self.target_model.load_weights(names[2])

    def update_target_model(self):
        self.target_model.set_weights(self.step_model.get_weights())

    def get_model(self, dueling=False):
        i = Input(self.state_shape)
        x = Conv2D(32, 8, (4, 4), activation='relu')(i)
        x = Conv2D(64, 4, (2, 2), activation='relu')(x)
        x = Conv2D(64, 3, (1, 1), activation='relu')(x)
        x = Flatten()(x)

        if dueling:
            a = Dense(512, activation='relu')(x)
            a = Dense(self.action_shape)(a)
            a = Lambda(lambda a: a[:, :] - tf.reduce_mean(a[:, :], axis=-1, keepdims=True),
                       output_shape=(self.action_shape,))(a)

            v = Dense(512, activation='relu')(x)
            v = Dense(1)(v)
            v = Lambda(lambda s: tf.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_shape,))(v)

            x = Add()([v, a])
        else:
            x = Dense(512, activation='relu')(x)
            x = Dense(self.action_shape)(x)

        return K.models.Model(inputs=i, outputs=x)

    def get_action(self, observation, explore=True):
        """
        get an action following an epsilon-greedy policy where an action is selected randomly with a probability of
        epsilon, else the action with the highest q value is selected from the model with current policy applied to
        the current observation.
        :param observation: the observation upon which the greedy action is based should it be take.
        array with shape = (*observation_space, remember_frames) is expected
        :param explore: if explore is True the agent will follow a greedy policy
        :return: None
        """
        a = 0

        if explore:
            if self.total_steps < self.train_start:  # to partially fill replay buffer with random experiences
                return self.env.action_space.sample()

            elif np.random.random() <= self.exploration_rate:  # explore the env with a probability of epsilon
                return self.env.action_space.sample()

            else:  # else take greedy action
                return self.get_action(observation, explore=False)

        else:  # greedy action
            a = self.model_qs_argmax(observation)

            return a.numpy()[0]

    @tf.function
    def model_qs_argmax(self, observation):
        observation = tf.math.divide(observation, 255)  # scale pixel values
        model_out = self.step_model(tf.expand_dims(observation, axis=0))  # get q values
        return tf.argmax(model_out, axis=-1) # return the action with the highest q value

    @tf.function
    def train_step(self, batch, scaling=True, double=False):
        """
        Train the Step model on a batch of (state, action, reward, next_state, terminal[aka done]) transitions
        :param
        batch: the batch of transition to train on must be in the format:
        [state_array, action_array, reward_array, next_step_array, terminal_array]
        :return:
        None
        """
        s0, a, r, s1, d = batch
        if scaling:  # WE LOVE AUTO GRAPH.... and its support for native python statements
            s0, s1 = s0 / 255, s1 / 255  # scale the pixel values
        a = tf.cast(a, dtype='int32')

        # creates a mask where 'done' is encoded as 0 and 'not done' is encoded as 1. this can then be multiplied
        # with the 'future reward term in the bellman equation ( y*(Q_target(s', a'=argmax(Q_step(s', a')) )
        # to mask off frames where the concept of a future reward is illogical nonsensical
        done_mask = tf.cast(tf.math.logical_not(d), dtype='float32')

        if double:
            # Q_step*(s,a)   <--   r + y * (Q_target(s', a'=argmax(Q_step(s', a')) DOUBLE BELLMAN
            pred_s1 = self.target_model(s1)  # Q_target(s', a') for all a'
            idxs = tf.expand_dims(tf.argmax(self.step_model(s1), axis=-1), axis=-1)  # a'=argmax(Q_step(s', a')
            s1_qs = tf.gather_nd(pred_s1, idxs, batch_dims=1)  # Q_target(s', a'=argmax(Q_step(s', a')
            act_ys = r + done_mask * (self.discount * s1_qs)  # r + y*(Q_target(s', a'=argmax(Q_step(s', a'))

        else:
            # Q_step*(s,a)   <--   r + y * max_a'(Q_target(s', a')) BELLMAN
            s1_qs = tf.reduce_max(self.target_model(s1), axis=-1)  # max_a'(Q_target(s', a'))
            act_ys = r + done_mask * (self.discount * s1_qs)  # r + y * max_a'(Q_target(s', a'))

        # gradient descent step, same as supervised learning
        with tf.GradientTape() as tape:
            pre_ys = tf.gather_nd(self.step_model(s0), tf.expand_dims(a, -1), batch_dims=1)
            loss = K.losses.mean_squared_error(act_ys, pre_ys)

        gradients = tape.gradient(loss, self.step_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.step_model.trainable_variables))

        self.model_loss(loss)


    def save_params(self, model_only=False):
        """
        save model parameters required to continue the training process
        :return: None
        """
        if not model_only:
            params = (
                self.exploration_decay,
                self.exploration_rate,
                self.total_steps,
                self.total_eps,
                self.total_train_steps,
                self.replay_memory,
            )
            if DUELING or DOUBLE:
                with open(f'agent_state_{GAME}.state', 'w+b') as f:
                    pickle.dump(params, f)
            else:
                with open(f'agent_state_simple_{GAME}.state', 'w+b') as f:
                    pickle.dump(params, f)

        if DUELING or DOUBLE:
            self.step_model.save_weights(f'step_model_simple_{GAME}.h5')
            self.target_model.save_weights(f'target_model_simple_{GAME}.h5')
        else:
            self.step_model.save_weights(f'step_model_simple_{GAME}.h5')
            self.target_model.save_weights(f'target_model_simple_{GAME}.h5')

    # TODO implement reset method for agent
    # TODO implement save agent params and state
    # TODO implement tensorboard

    def run_ep(self, ep_idx=None, save_metrics=False):
        ep_start = time.time()
        done = False
        step = 0
        ep_reward = 0
        state = self.env.reset()

        while not done and step < 100_000:
            # t0 = time.time()
            action = self.get_action(state)  # get action
            state, reward, done, info = self.env.step(action)  # make step with chosen action

            # t1 = time.time()
            self.replay_memory.push((state, action, reward, done))  # update replay memory
            ep_reward += reward
            # t2 = time.time()

            # start training after minimum experiences in memory
            if self.total_steps >= self.train_start and self.total_steps % self.optimizer_frequency == 0:
                self.train_step(self.replay_memory.minibatch(), double=DOUBLE)

            # t3 = time.time()
            # update target every n steps
            if self.total_steps % self.target_update == 0:
                self.update_target_model()

            # the exploration rate will not decay until start of training
            if self.total_steps == self.train_start:
                self.exploration_decay = True

            # manage the decay of the exploration rate
            if self.exploration_decay:
                if EXPONENTIAL_EXPLORATION_DECAY:
                    self.exploration_rate *= self.exploration_factor
                else:
                    self.exploration_rate += self.exploration_slope

                # turn off exploration rate after n steps
                if self.total_steps - self.train_start >= self.exploration_frames_to_final:
                    self.exploration_decay = False

            # print(time.time()-t0, t1-t0, t2-t1, t3-t2)

            # update counters
            step += 1
            self.total_steps += 1

        # saving metrics
        if save_metrics:
            assert ep_idx is not None, 'no episode indexing given while summary is enabled'
            with self.stats_summary_writer.as_default():
                tf.summary.scalar('loss', self.model_loss.result(), self.total_eps)
                tf.summary.scalar('ep_reward', ep_reward, self.total_eps)
                tf.summary.scalar('ep_steps', step, self.total_eps)
                tf.summary.scalar('time per ep', time.time()-ep_start, self.total_eps)
                tf.summary.scalar('current exploration rate', self.exploration_rate, self.total_eps)
            self.ep_losses.append(self.model_loss.result())

        # update counters
        self.steps_per_ep.append(step)
        self.ep_rewards.append(ep_reward)
        self.total_eps += 1


if __name__ == "__main__":
    agent = Agent(load_from_file=False)
    agent.save_params(model_only=True)
    agent.step_model.save(f'full_models/step_model_{GAME}.h5')
    agent.step_model.save(f'full_models/target_model_{GAME}.h5')
    e = 0
    while agent.total_steps <= 100_000_000:
        agent.run_ep(e, save_metrics=True)
        if e % 100 == 0:
            agent.save_params(model_only=True)
            print(f'loss moving avg past 10: {np.mean(agent.ep_losses[-10:])} '
                  f'ep_reward moving avg past 10: {np.mean(agent.ep_rewards[-10:])} \n'
                  f'current exploration rate: {agent.exploration_rate}')

        if e % 500 == 0:
            agent.ckpt_manager.save()
            agent.step_model.save(f'full_models/step_model_{GAME}.h5')
            agent.step_model.save(f'full_models/target_model_{GAME}.h5')

        e += 1

