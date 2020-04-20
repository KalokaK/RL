import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pickle
from gym.wrappers.cusotm_atari_preprocessing import AtariPreprocessing
import gym.envs.atari.atari_env
from replay_memory import Storage
import matplotlib.pyplot as plt
from helpers import measure
import os, datetime
import helpers

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
START_TRAINING_AFTER = 10_000
FRAMES_TO_INCLUDE = 4
UPDATE_TARGET_EVERY = 1_000
DISCOUNT = 0.99
ACTION_REPEAT = 4  # ie. agent sees every n th frame
OPTIMIZER_STEP_EVERY = 4  # optimizer is applied every n th action choice
LEARN_RATE = 0.00025
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.07
FRAMES_TO_REACH_FINAL = 1_000_000
MAX_NO_ACTION = 30
GAME = 'pong'

# utility objects
env = gym.envs.atari.atari_env.AtariEnv(game=GAME, obs_type='image', frameskip=1)


class Agent:
    def __init__(self, load_from_file=False):
        # params for training
        self.env = AtariPreprocessing(env, noop_max=MAX_NO_ACTION)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n

        self.total_steps = 0
        self.total_eps = 0
        self.total_train_steps = 0
        self.reward_per_ep = []
        self.steps_per_ep = []
        self.ep_losses = []

    def get_action(self, observation):
        pass

    def save_params(self):
        pass

    def step(self):
        pass

    def run_ep(self, ep_idx=None, save_metrics=False):
        done = False
        step = 0
        state = self.env.reset()
        while not done and step < 100_000:
            self.step()

            # update counters
            step += 1
            self.total_steps += 1

        # update counters
        self.steps_per_ep.append(step)
        self.total_eps += 1
