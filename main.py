import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Add, Lambda, Input
import os, datetime
import helpers
from gym.wrappers.cusotm_atari_preprocessing import AtariPreprocessing
import gym.envs.atari.atari_env
from replay_memory import Storage
from helpers import measure

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
MAX_REPLAY_MEMORY_SIZE = 10_000
START_TRAINING_AFTER = 1_000
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
GAME = 'breakout'

# utility objects
env = gym.envs.atari.atari_env.AtariEnv(game=GAME, obs_type='image', frameskip=1)
optimizer = K.optimizers.Adam(LEARN_RATE)


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

        self.exploration_rates = np.linspace(EXPLORATION_RATE, MIN_EXPLORATION_RATE, FRAMES_TO_REACH_FINAL)
        self.exploration_decay = False
        self.exploration_idx = 0
        self.min_exploration_rate = MIN_EXPLORATION_RATE
        self.exploration_frames_to_final = FRAMES_TO_REACH_FINAL

        self.env = AtariPreprocessing(env, noop_max=MAX_NO_ACTION)
        self.state_shape = (*self.env.observation_space.shape, self.remember_frames)
        self.action_shape = self.env.action_space.n

        # models and other objects
        self.step_model = self.get_model()
        self.target_model = self.get_model()
        self.replay_memory = Storage(self.max_replay, self.state_shape, self.minibatch_size)
        self.frame_buffer = helpers.ObservationBuffer(self.env.reset(), self.remember_frames)

        # metrics
        self.model_loss = tf.keras.metrics.Mean('model_loss', dtype=tf.float32)
        self.model_accuracy = K.metrics.MeanSquaredError('model_accuracy')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_log_dir = 'logs/gradient_tape/' + current_time
        self.model_summary_writer = tf.summary.create_file_writer(model_log_dir)

        self.total_steps = 0
        self.total_eps = 0
        self.total_train_steps = 0
        self.steps_per_ep = []
        self.ep_losses = []
        self.ep_accuracies = []

        self.step_model.summary()
        self.target_model.summary()

        if load_from_file:
            with open('agent_state.state', 'rb') as f:
                (
                    self.exploration_decay,
                    self.exploration_idx,
                    self.total_steps,
                    self.total_eps,
                    self.total_train_steps,
                    self.replay_memory
                ) = pickle.load(f)

            self.step_model.load_weights('step_model.h5')
            self.target_model.load_weights('target_model.h5')

    def get_model(self):
        i = Input(self.state_shape)
        x = Conv2D(32, 8, (4, 4), activation='relu')(i)
        x = Conv2D(64, 4, (2, 2), activation='relu')(x)
        x = Conv2D(64, 3, (1, 1), activation='relu')(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(self.action_shape)(x)

        return K.models.Model(inputs=i, outputs=x)

    def update_target_model(self):
        self.target_model.set_weights(self.step_model.get_weights())

    def get_action(self, observation):
        a = 0
        if self.exploration_decay or self.total_steps <= self.train_start:
            if self.exploration_rates[self.exploration_idx] >= np.random.random():
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
        s0, s1 = tf.math.divide(s0, 255), tf.math.divide(s1, 255)
        r = tf.expand_dims(r, -1)
        s1_max_qs = tf.reduce_max(self.target_model(s1), keepdims=True, axis=-1)
        a = tf.one_hot(a, self.action_shape, axis=-1)
        done_mask = tf.expand_dims(tf.cast(tf.math.logical_not(d), dtype='float32'), 1)
        act_ys = a * (r + done_mask * (self.discount * s1_max_qs))

        with tf.GradientTape() as tape:
            pre_ys = self.step_model(s0)
            loss = K.losses.mean_squared_error(act_ys, pre_ys)

        gradients = tape.gradient(loss, self.step_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.step_model.trainable_variables))

        self.model_loss(loss)
        self.model_accuracy(act_ys, pre_ys)

    def save_params(self):
        params = (
            self.exploration_decay,
            self.exploration_idx,
            self.total_steps,
            self.total_eps,
            self.total_train_steps,
            self.replay_memory
        )
        with open('agent_state.state', 'w+b') as f:
            pickle.dump(params, f)

        self.step_model.save_weights('step_model.h5')
        self.target_model.save_weights('target_model.h5')

    # TODO implement reset method for agent
    # TODO implement save agent params and state
    # TODO implement tensorboard

    def run_ep(self, ep_idx=None, save_metrics=False):
        done = False
        step = 0
        state = self.env.reset()
        self.frame_buffer.reset(state)
        while not done and step < 100_000:
            action = self.get_action(self.frame_buffer.buffer)  # get action
            # print(f'action: {action}')
            _, reward, done, info = self.env.step(action)  # make step with chosen action
            state = self.frame_buffer.push(_)  # update frame buffer
            self.replay_memory.push((state, action, reward, done))  # update replay memory

            self.env.render()

            if self.total_steps >= self.train_start:  # start training after minimum experiences in memory
                self.train_step(self.replay_memory.minibatch())

            if self.total_steps % self.target_update:  # update target every n steps
                self.update_target_model()

            if self.total_steps == self.train_start:  # the exploration rate will not decay until start of training
                self.exploration_decay = True

            if self.exploration_decay:  # manage the decay of the exploration rate
                self.exploration_idx += 1
                if self.exploration_idx >= self.exploration_frames_to_final:  # turn off exploration rate after n steps
                    self.exploration_decay = False

            # update counters
            step += 1
            self.total_steps += 1

        # saving metrics
        if save_metrics:
            assert ep_idx is not None, 'no episode indexing given while summary is enabled'
            with self.model_summary_writer.as_default():
                tf.summary.scalar('loss', self.model_loss.result(), ep_idx)
                tf.summary.scalar('accuracy', self.model_accuracy.result(), ep_idx)
            self.ep_losses.append(self.model_loss.result())
            self.ep_accuracies.append(self.model_accuracy.result())

        # update counters
        self.steps_per_ep.append(step)
        self.total_eps += 1


if __name__ == "__main__":
    agent = Agent()
    agent.save_params()
    for e in range(20000):
        agent.run_ep(e, save_metrics=True)
        print(f'loss: {agent.model_loss.result()}, accuracy: {agent.model_accuracy.result()}')
        if e % 100 == 0:
            agent.save_params()
            print(f'loss: {agent.model_loss.result()}, accuracy: {agent.model_accuracy.result()}\n'
                  f'loss moving avg past 10: {np.mean(agent.ep_losses[-10:])} '
                  f'accuracy moving avg past 10: {np.mean(agent.ep_accuracies[-10:])} \n'
                  f'loss moving avg past 10: {np.mean(agent.ep_losses[-100:])} '
                  f'accuracy moving avg past 50: {np.mean(agent.ep_accuracies[-50:])}')
