# Agent parameters
EXPLORATION_RATE = 0.1
MIN_EXPLORATION_RATE = 0.1
FRAMES_TO_REACH_FINAL = 500_000
EXPONENTIAL_EXPLORATION_DECAY = False  # default is linear
MINIBATCH_SIZE = 32
MAX_REPLAY_MEMORY_SIZE = 200_00
START_TRAINING_AFTER = 10_000
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


https://www.youtube.com/watch?v=m7JMlE1SfPc boxing video
