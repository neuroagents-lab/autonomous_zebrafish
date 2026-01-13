import numpy as np
import pickle
from typing import Any

class RewardNormalizer:
    """
    Tracks running statistics (mean and variance) for undiscounted returns using Welford's online algorithm.
    Maintains separate statistics for each environment.
    """
    
    def __init__(self, num_envs, epsilon=1e-8):
        self.num_envs = num_envs
        self.epsilon = epsilon
        self.count = np.zeros(num_envs)
        self.mean = np.zeros(num_envs)
        self.M2 = np.zeros(num_envs)  # Sum of squares of differences from the mean

    def update(self, rewards):
        for i in range(self.num_envs):
            self.count[i] += 1
            delta = rewards[i] - self.mean[i]
            self.mean[i] += delta / self.count[i]
            delta2 = rewards[i] - self.mean[i]
            self.M2[i] += delta * delta2

    def get_std(self):
        # Compute sample variance; use epsilon to avoid division by zero.
        variance = np.zeros_like(self.count)
        mask = self.count > 1
        variance[mask] = self.M2[mask] / (self.count[mask] - 1)
        return np.sqrt(variance) + self.epsilon

    def update_and_get_std(self, rewards):
        self.update(rewards)
        return self.get_std()

class MovingAverage(object):
  """Computes the moving average of a variable."""

  def __init__(self, capacity):
    self._capacity = capacity
    self._history = np.empty(capacity, dtype=object)
    self._size = 0

  def add(self, value):
    index = self._size % self._capacity
    self._history[index] = value
    self._size += 1

  def mean(self, axis=0):
    if not self._size:
      return None
    if self._size < self._capacity:
      return np.mean(self._history[0:self._size], axis=axis)
    return np.mean(self._history, axis=axis)
  
  def std(self, axis=0):
    if not self._size:
      return None
    
    if self._size < self._capacity:
      return np.std(self._history[0:self._size], axis=axis)
    
    return np.std(self._history, axis=axis)

  def _last_n_slice(self, n):
      """Return a view of the last n entries (in insertion order)."""
      if self._size == 0:
          return np.empty((0,), dtype=object)
      n = min(n, self._size, self._capacity)
      if self._size <= self._capacity:
          return self._history[self._size - n:self._size]
      # circular buffer has wrapped
      start = (self._size - n) % self._capacity
      end = (start + n)
      if end <= self._capacity:
          return self._history[start:end]
      # wraps around the end of the array
      return np.concatenate([
          self._history[start:],
          self._history[: end % self._capacity]
      ])

  def mean_last_n(self, n, axis=0):
      """
      Mean over the last n added values.
      """
      data = self._last_n_slice(n)
      if data.size == 0:
          return None
      return np.mean(data, axis=axis)

  def std_last_n(self, n, axis=0):
      """
      Standard deviation over the last n added values.
      """
      data = self._last_n_slice(n)
      if data.size == 0:
          return None
      return np.std(data, axis=axis)
