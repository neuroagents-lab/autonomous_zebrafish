import os
import random
from typing import Any, Optional

from torch import nn
import torch
import numpy as np
from tqdm import tqdm

class ProgressTrainer:
    """Update the Learning Progress network in an online way."""

    def __init__(self,
                 progress_module,
                 observation_history_size=2000,
                 training_interval=100,
                 num_epochs=3,
                 batch_size=50,
                 **kwargs : Any):
        assert training_interval > 0, "Training interval must be greater than 0."

        self.progress_module = progress_module
        self.current_world_model = progress_module.current_world_model
        self.trailing_world_model = progress_module.trailing_world_model

        self._opt = self.current_world_model.opt

        self._wm_training_interval = training_interval
        self._observation_history_size = observation_history_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        # Keeps track of the last N observations.
        # need access to feature and action space dim
        self._fifo_observations = [np.empty(320)] * observation_history_size
        self._fifo_actions = [np.empty(5)] * observation_history_size
        self._fifo_next_observations = [np.empty(320)] * observation_history_size
        #self._fifo_dones = [None] * observation_history_size
        self._fifo_index = 0
        self._fifo_count = 0

    def on_new_observation(self, obs, actions, next_obs, dones, infos):
        """Event triggered when the environments generate a new observation."""
        self._fifo_observations[self._fifo_index] = obs
        self._fifo_next_observations[self._fifo_index] = next_obs
        self._fifo_actions[self._fifo_index] = actions
        #self._fifo_dones[self._fifo_index] = dones
        self._fifo_index = ((self._fifo_index + 1) % self._observation_history_size)
        self._fifo_count += 1

        if self._fifo_count > 0 and self._fifo_count % self._wm_training_interval == 0:
            print('Training the World Model after: {}'.format(self._fifo_count))

            history_observations, history_actions, history_next_observations = self._get_flatten_history()
            self.train(history_observations, history_actions, history_next_observations)
            # update the trailing world model <- I think this is where update should be called? 
            self.progress_module.update()

    def _get_flatten_history(self):
        """Convert the history given as a circular fifo to a linear array."""
        if self._fifo_count < len(self._fifo_observations):
         return (self._fifo_observations[:self._fifo_count],
                self._fifo_actions[:self._fifo_count],
                self._fifo_next_observations[:self._fifo_count],)
                #self._fifo_dones[:self._fifo_count])

        # Reorder the indices.
        history_observations = self._fifo_observations[self._fifo_index:]
        history_observations.extend(self._fifo_observations[:self._fifo_index])
        history_actions = self._fifo_actions[self._fifo_index:]
        history_actions.extend(self._fifo_actions[:self._fifo_index])
        history_next_observations = self._fifo_next_observations[self._fifo_index:]
        history_next_observations.extend(self._fifo_next_observations[:self._fifo_index])
        #history_dones = self._fifo_dones[self._fifo_index:]
        #history_dones.extend(self._fifo_dones[:self._fifo_index])
        return history_observations, history_actions, history_next_observations #, history_dones
    
    def train(self, obs, actions, next_obs):
        """Do one pass of training of the World Model."""    
        self.fit(
            self._generate_batch(obs, actions, next_obs),
            steps_per_epoch = self._observation_history_size // self._batch_size,
            epochs=self._num_epochs
            )

    def fit(self, gen, steps_per_epoch, epochs):
        for step in range(steps_per_epoch * epochs):
            obs, actions, next_obs = next(gen)
            world_model_pred = self.current_world_model(obs, actions)
            world_model_loss = self.current_world_model.loss(world_model_pred, next_obs)
            self._opt.zero_grad()
            world_model_loss.backward()
            self._opt.step()

    def _generate_batch(self, obs, actions, next_obs):
        """Generate batches of data used to train the ICM."""
        sample_count = len(actions)
        while True:
            number_of_batches = sample_count // self._batch_size
            for batch_index in range(number_of_batches):
                from_index = batch_index * self._batch_size
                to_index = (batch_index + 1) * self._batch_size
                yield (np.array(obs[from_index:to_index]),
                        np.array(actions[from_index:to_index]),
                        np.array(next_obs[from_index:to_index]))
