import numpy as np
from _collections import deque
from helpers import measure

class Storage(object):

    def __init__(self, max_len: int, state_shape: tuple, minibatch_size: int = 32):
        """
        Stores state - action transitions from one state to the next. based on those training is
        later performed on minibatches.
        :param max_len:
        the maximum amount of transitions the replay buffer can hold
        :param state_shape:
        the shape of the actual state information array.
        :param minibatch_size:
        how large the batches which are going to be drawn from the buffer are supposed to be
        """
        self.max_len = max_len
        self.state_shape = state_shape
        self.minibatch_size = minibatch_size
        self.frame_storage = deque(np.empty((max_len, *state_shape), dtype='uint8'), max_len)
        self.action_storage = deque(np.empty(max_len, dtype='uint8'), max_len)
        self.reward_storage = deque(np.empty(max_len, dtype='float32'), max_len)
        self.terminal_storage = deque(np.empty(max_len, dtype=bool), max_len)

        self.s0 = np.empty((32, *self.state_shape), dtype='float32')
        self.a = np.empty(32, dtype='uint8')
        self.r = np.empty(32, dtype='float32')
        self.s1 = np.empty((32, *self.state_shape), dtype='float32')
        self.t = np.empty(32, dtype='bool')

    # @measure
    def push(self, experience: tuple):
        """
        append one state action transition to the replay buffer.
        :param experience:
        in the format of (state: array, action: int, reward: float, terminal: bool)
        :return:
        None
        """
        self.frame_storage.appendleft(experience[0])
        self.action_storage.appendleft(experience[1])
        self.reward_storage.appendleft(experience[2])
        self.terminal_storage.appendleft(experience[3])

    # @measure
    def minibatch(self):
        """
        Gets a minibatch of the size passed to the __init__ method.
        :return:
        returns arrays of matching before-states (s0), actions (a), rewards (r),
        after-states(s1), and terminals (t)
        """
        idxs = np.random.randint(0, self.max_len - 2, self.minibatch_size)

        for i in range(self.minibatch_size):
            idx = idxs[i]
            if self.terminal_storage[idx]:
                idx += 1

            self.s0[i] = self.frame_storage[idx + 1]
            self.a[i] = self.action_storage[idx]
            self.r[i] = self.reward_storage[idx]
            self.s1[i] = self.frame_storage[idx]
            self.t[i] = self.terminal_storage[idx]

        return self.s0, self.a, self.r, self.s1, self.t


