
from functools import wraps
from time import time
import numpy as np

# courtesy of https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837

def measure(func):
    """
    courtesy of https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837

    a simple timing wrapper since I absolutely hate the timeit library
    :param func:
    the function to be timed
    :return:
    the function value
    """
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time() * 1000
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time() * 1000 - start
            print(f"Total execution time for {func.__name__}: {end_ if end_ > 0 else 0} ms")
    return _time_it


class ObservationBuffer:
    """
    a neat way to stack frames
    """
    def __init__(self, observation: np.ndarray, frames: int):
        """
        a frame buffer to stack frames for training with temporal context
        :param observation:
        an initial observation with which the frame buffer is filled
        :param frames:
        must be bigger than 0, the amount of past frames to stack
        """
        assert frames > 0, 'frames but be bigger than 0'
        self.frames = frames
        self.buffer = np.stack([observation]*frames, axis=-1)

    def push(self, observation: np.ndarray):
        """
        adds an image to the current observation buffer and returns the buffer + the last n images, where n is defined
        in the __init__ function
        :param observation:
        the observation to be added to the buffer,
        :return:
        the observation buffer after the observation has been added
        """
        assert self.buffer[:, :, 0].shape == observation.shape, 'invalid observation shape'

        self.buffer = np.roll(self.buffer, 1, axis=-1)
        self.buffer[:, :, 0] = observation
        return self.buffer

    def reset(self, observation):
        assert self.buffer[:, :, 0].shape == observation.shape, 'invalid observation shape'

        self.buffer = np.stack([observation]*self.frames, axis=-1)
