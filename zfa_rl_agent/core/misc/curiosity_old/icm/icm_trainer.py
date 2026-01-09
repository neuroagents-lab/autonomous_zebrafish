import os
import random
from typing import Any, Optional

from torch import nn
import torch
import numpy as np
from tqdm import tqdm

LR = 1e-4

class ICMTrainer:
    """Train an ICM network in an online way."""

    def __init__(self,
                 icm_model,
                 observation_history_size=1000,
                 training_interval=100,
                 num_epochs=5,
                 batch_size=64,
                 **kwargs : Any):
        if training_interval < 0:
            training_interval = observation_history_size

        self._icm_model = icm_model
        self._opt = torch.optim.Adam(icm_model.parameters(), lr=LR)

        self._training_interval = training_interval
        self._observation_history_size = observation_history_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        # Keeps track of the last N observations.
        # Those are used to train the ICM network in an online way.
        self._fifo_observations = [None] * observation_history_size
        self._fifo_actions = [None] * observation_history_size
        self._fifo_next_observations = [None] * observation_history_size
        self._fifo_dones = [None] * observation_history_size
        self._fifo_index = 0
        self._fifo_count = 0

        self._current_epoch = 0

    def on_new_observation(self, obs, actions, next_obs, dones, infos):
        """Event triggered when the environments generate a new observation."""
        self._fifo_observations[self._fifo_index] = obs
        self._fifo_next_observations[self._fifo_index] = next_obs
        self._fifo_actions[self._fifo_index] = actions
        self._fifo_dones[self._fifo_index] = dones
        self._fifo_index = (
                (self._fifo_index + 1) % self._observation_history_size)
        self._fifo_count += 1

        if self._fifo_count > 0 and self._fifo_count % self._training_interval == 0:
            print('Training the ICM after: {}'.format(self._fifo_count))
            history_observations, history_actions, history_next_observations, history_dones = self._get_flatten_history()
            self.train(history_observations, history_actions, history_next_observations, history_dones)

    def _get_flatten_history(self):
        """Convert the history given as a circular fifo to a linear array."""
        if self._fifo_count < len(self._fifo_observations):
            return (self._fifo_observations[:self._fifo_count],
                    self._fifo_actions[:self._fifo_count],
                    self._fifo_next_observations[:self._fifo_count],
                    self._fifo_dones[:self._fifo_count])

        # Reorder the indices.
        history_observations = self._fifo_observations[self._fifo_index:]
        history_observations.extend(self._fifo_observations[:self._fifo_index])
        history_actions = self._fifo_actions[self._fifo_index:]
        history_actions.extend(self._fifo_actions[:self._fifo_index])
        history_next_observations = self._fifo_next_observations[self._fifo_index:]
        history_next_observations.extend(self._fifo_next_observations[:self._fifo_index])
        history_dones = self._fifo_dones[self._fifo_index:]
        history_dones.extend(self._fifo_dones[:self._fifo_index])
        return history_observations, history_actions, history_next_observations, history_dones
    
    def train(self, obs, actions, next_obs, dones):
        """Do one pass of training of the ICM."""
        #for i in range(len(obs['pixels'])): print("train", obs['pixels'][i].shape) 
    
        self.fit(
            self._generate_batch(obs, actions, next_obs),
            steps_per_epoch = self._observation_history_size // self._batch_size,
            epochs=self._num_epochs
            )

    def fit(self, gen, steps_per_epoch, epochs):
        for step in range(steps_per_epoch * epochs):
            obs, actions, next_obs = next(gen)
            print(obs.shape)
            forward_pred_error, inverse_pred_error = self._icm_model(obs, actions, next_obs)
            loss = self._icm_model.loss_fn(forward_pred_error, inverse_pred_error)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

    def _generate_batch(self, obs, actions, next_obs):
        """Generate batches of data used to train the ICM."""
        sample_count = len(obs)
        while True:
            number_of_batches = sample_count // self._batch_size
            for batch_index in range(number_of_batches):
                from_index = batch_index * self._batch_size
                to_index = (batch_index + 1) * self._batch_size
                obs_batch = {key: np.array(obs[key])[from_index:to_index] for key in self.obs_keys}
                next_obs_batch = {key: np.array(next_obs[key])[from_index:to_index]for key in self.obs_keys}
                actions_batch = np.array(actions)[from_index:to_index]

                yield obs_batch, actions_batch, next_obs_batch
