import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Add, Lambda, Input
import os, datetime, time
import helpers
from gym.wrappers.cusotm_atari_preprocessing import AtariPreprocessing
import gym.envs.atari.atari_env
from replay_memory import Storage
from helpers import measure

plt.ion()

GPU = True
if not GPU: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# Hpyerparameters
MINIBATCH_SIZE = 32
MAX_REPLAY_MEMORY_SIZE = 50_000
START_TRAINING_AFTER = 10_000
FRAMES_TO_INCLUDE = 4
UPDATE_TARGET_EVERY = 10_000
DISCOUNT = 0.99
ACTION_REPEAT = 4  # ie. agent sees every n th frame
OPTIMIZER_STEP_EVERY = 4  # optimizer is applied every n th action choice
LEARN_RATE = 0.00025
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.1
FRAMES_TO_REACH_FINAL = 1_000_000
MAX_NO_ACTION = 30
GAME = 'boxing'

# utility objects
env = gym.envs.atari.atari_env.AtariEnv(game=GAME, obs_type='image', frameskip=1)
optimizer = K.optimizers.Adam(LEARN_RATE, beta_1=0.95, beta_2=0.95, epsilon=0.01)


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

        self.exploration_slope = (MIN_EXPLORATION_RATE - EXPLORATION_RATE) / FRAMES_TO_REACH_FINAL
        self.exploration_decay = False
        self.exploration_rate = 1.0
        self.min_exploration_rate = MIN_EXPLORATION_RATE
        self.exploration_frames_to_final = FRAMES_TO_REACH_FINAL

        self.env = AtariPreprocessing(env, noop_max=MAX_NO_ACTION)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n

        # models and other objects
        self.step_model = self.get_model()
        self.target_model = self.get_model()
        self.update_target_model()
        self.replay_memory = Storage(self.max_replay, self.state_shape, self.minibatch_size)
        self.ckpt = tf.train.Checkpoint(model=self.step_model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, './checkpoint', max_to_keep=10)

        # metrics
        self.model_loss = tf.keras.metrics.Mean('model_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_log_dir = 'logs/gradient_tape/' + current_time
        self.model_summary_writer = tf.summary.create_file_writer(model_log_dir)

        self.total_steps = 0
        self.total_eps = 0
        self.total_train_steps = 0
        self.steps_per_ep = []
        self.ep_losses = []
        self.ep_rewards = []

        self.step_model.summary()
        self.target_model.summary()

        if load_from_file:
            with open('agent_state.state', 'rb') as f:
                params = pickle.load(f)
                self.exploration_decay = params[0]
                self.exploration_rate = params[1]
                self.total_steps = params[2]
                self.total_eps = params[3]
                self.total_train_steps = params[4]
                self.replay_memory = params[5]

            self.step_model.load_weights('step_model.h5')
            self.target_model.load_weights('target_model.h5')

    def get_model(self):
        i = Input(self.state_shape)
        x = Conv2D(32, 8, (4, 4), activation='relu')(i)
        x = Conv2D(64, 4, (2, 2), activation='relu')(x)
        x = Conv2D(64, 3, (1, 1), activation='relu')(x)
        x = Flatten()(x)

        a = Dense(512, activation='relu')(x)
        a = Dense(self.action_shape)(a)
        a = Lambda(lambda a: a[:, :] - tf.reduce_mean(a[:, :], axis=-1, keepdims=True),
                   output_shape=(self.action_shape,))(a)

        v = Dense(512, activation='relu')(x)
        v = Dense(1)(v)
        v = Lambda(lambda s: tf.expand_dims(s[:, 0], -1),
                   output_shape=(self.action_shape,))(v)

        x = Add()([v, a])

        return K.models.Model(inputs=i, outputs=x)

    def update_target_model(self):
        self.target_model.set_weights(self.step_model.get_weights())

    def get_action(self, observation):
        a = 0
        if self.exploration_decay or self.total_steps <= self.train_start:
            if self.exploration_rate >= np.random.random():
                a = self.env.action_space.sample()
        elif self.min_exploration_rate >= np.random.random():
            a = self.env.action_space.sample()

        else:
            observation = tf.math.divide(observation, 255)
            model_out = self.step_model(tf.expand_dims(observation, axis=0))
            a = tf.argmax(model_out, axis=-1).numpy()[0]

        return a

    @tf.function
    def train_step(self, batch):
        s0, a, r, s1, d = batch
        s0, s1 = s0 / 255, s1 / 255  # scale the pixel values
        r = tf.expand_dims(r, -1)
        a = tf.one_hot(a, self.action_shape, axis=-1)
        done_mask = tf.expand_dims(tf.cast(tf.math.logical_not(d), dtype='float32'), 1)

        # Q_step*(s,a)   <--   r + y*(Q_step(s', a'=argmax(Q_target(s', a')) DOUBLE DQN BELLMAN
        pred_s1 = self.step_model(s1)  # Q_step(s', a') for all a'
        idxs = tf.expand_dims(tf.argmax(self.target_model(s1), axis=-1), axis=-1)  # a'=argmax(Q_target(s', a')
        s1_qs = tf.expand_dims(tf.gather_nd(pred_s1, idxs, batch_dims=1), -1)  # Q_step(s', a'=argmax(Q_target(s', a')
        act_ys = a * (r + done_mask * (self.discount * s1_qs))  # r + y*(Q_step(s', a'=argmax(Q_target(s', a'))

        with tf.GradientTape() as tape:
            pre_ys = self.step_model(s0)
            loss = K.losses.mean_squared_error(act_ys, pre_ys)

        gradients = tape.gradient(loss, self.step_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.step_model.trainable_variables))

        self.model_loss(loss)

    def save_params(self):
        params = (
            self.exploration_decay,
            self.exploration_rate,
            self.total_steps,
            self.total_eps,
            self.total_train_steps,
            self.replay_memory,
        )
        with open('agent_state.state', 'w+b') as f:
            pickle.dump(params, f)

        self.step_model.save_weights('step_model.h5')
        self.target_model.save_weights('target_model.h5')

    # TODO implement reset method for agent
    # TODO implement save agent params and state
    # TODO implement tensorboard

    def run_ep(self, ep_idx=None, save_metrics=False):
        ep_start = time.time()
        done = False
        step = 0
        ep_reward = 0
        state = self.env.reset()
        plt.imsave('img.png', state[:, :, 0])
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
                self.train_step(self.replay_memory.minibatch())
            # t3 = time.time()
            # update target every n steps
            if self.total_steps % self.target_update == 0:
                self.update_target_model()
            # the exploration rate will not decay until start of training
            if self.total_steps == self.train_start:
                self.exploration_decay = True
            # manage the decay of the exploration rate
            if self.exploration_decay:
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
            with self.model_summary_writer.as_default():
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
    agent.save_params()
    for e in range(20000):
        agent.run_ep(e, save_metrics=True)
        if e % 10 == 0:
            agent.save_params()
            print(f'loss moving avg past 10: {np.mean(agent.ep_losses[-10:])} '
                  f'ep_reward moving avg past 10: {np.mean(agent.ep_rewards[-10:])} \n'
                  f'current exploration rate: {agent.exploration_rate}')

        if e % 50 == 0:
            agent.ckpt_manager.save()

