from typing import Any, Optional
import numpy as np
import torch 

class CuriosityTrainer:
    """Online world model training."""

    def __init__(self,
                 curiosity_module,
                 observation_history_size=100, # ~ one episode of 1000 steps x 64 envs
                 training_interval=100, # ~ 100 steps x 64
                 num_epochs=1,
                 batch_size=1,
                 reset_memory = False,
                 ):
        assert training_interval > 0, "Training interval must be greater than 0."
        
        self.name = curiosity_module.name + "_trainer"
        self.curiosity_module = curiosity_module
        self.world_model = curiosity_module.world_model
        self._opt = curiosity_module.opt

        if hasattr(self.curiosity_module, 'ensemble_size'):
            self.ensemble_size = self.curiosity_module.ensemble_size
        else:
            self.ensemble_size = 1

        if hasattr(self.curiosity_module, 'beta'):  
            self.beta = self.curiosity_module.beta
        else:
            self.beta = 1.0

        self._wm_training_interval = training_interval
        self._observation_history_size = observation_history_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        # Keeps track of the last N observations.
        self.obs_size = curiosity_module.feature_size
        self.action_size = curiosity_module.action_size
        self._fifo_observations = [np.empty(self.obs_size)] * observation_history_size
        self._fifo_actions = [np.empty(self.action_size)] * observation_history_size
        self._fifo_next_observations = [np.empty(self.obs_size)] * observation_history_size
        self._fifo_index = 0
        self._fifo_count = 0

        # whether to reset the memory after the fifo buffer is filled
        self.reset_memory = reset_memory

    def on_new_observation(self, obs, actions, next_obs, dones, infos):
        """Event triggered when the environments generate a new observation."""
        self._fifo_observations[self._fifo_index] = obs
        self._fifo_next_observations[self._fifo_index] = next_obs
        self._fifo_actions[self._fifo_index] = actions
        #self._fifo_dones[self._fifo_index] = dones
        #self._fifo_infos[self._fifo_index] = infos
        self._fifo_index = ((self._fifo_index + 1) % self._observation_history_size)
        self._fifo_count += 1
        reset = self._fifo_count == self._observation_history_size and self.reset_memory
        if self._fifo_count > 0 and self._fifo_count % self._wm_training_interval == 0:
            print(f"Training world model with {self._fifo_count} observations")
            history_observations, history_actions, history_next_observations = self._get_flatten_history()
            self.train(history_observations, history_actions, history_next_observations)
            self.curiosity_module.update(history_observations, history_actions, history_next_observations)


        if reset:
            # reset fifo buffer
            print(f"Resetting memory after {self._fifo_count} observations")
            self._fifo_count = 0
            self._fifo_index = 0
            self._fifo_observations = [np.empty(self.obs_size)] * self._observation_history_size
            self._fifo_actions = [np.empty(self.action_size)] * self._observation_history_size
            self._fifo_next_observations = [np.empty(self.obs_size)] * self._observation_history_size

            # reset world models
            self.curiosity_module.reset()

    def _get_flatten_history(self):
        """Convert the history given as a circular fifo to a linear array."""
        if self._fifo_count < len(self._fifo_observations):
            obs_list = self._fifo_observations[:self._fifo_count]
            act_list = self._fifo_actions[:self._fifo_count]
            next_list = self._fifo_next_observations[:self._fifo_count]
        else:
            obs_list = self._fifo_observations[self._fifo_index:] + \
                          self._fifo_observations[:self._fifo_index]
            act_list = self._fifo_actions[self._fifo_index:] + \
                          self._fifo_actions[:self._fifo_index]
            next_list = self._fifo_next_observations[self._fifo_index:] + \
                          self._fifo_next_observations[:self._fifo_index]

        # now convert into 1 big Tensor each
        # if entries are numpy arrays this will auto‐convert them
        
        obs_tensor = torch.stack(obs_list, dim=0)
        actions_tensor = torch.stack(act_list, dim=0)
        next_tensor = torch.stack(next_list, dim=0)

        return obs_tensor, actions_tensor, next_tensor
    
    def train(self, obs, actions, next_obs):
        """Do one pass of training of the World Model."""    
        for i in range(self.ensemble_size):
            # consider not shuffling the data? 
            rng_state = torch.random.get_rng_state()
            obs = obs[torch.randperm(obs.size(0))]
            torch.random.set_rng_state(rng_state)
            actions = actions[torch.randperm(actions.size(0))]
            torch.random.set_rng_state(rng_state)
            next_obs = next_obs[torch.randperm(next_obs.size(0))]
            self.fit(
                self._generate_batch(obs, actions, next_obs),
                steps_per_epoch = self._observation_history_size // self._batch_size,
                epochs=self._num_epochs,
                ensemble_index=i)

    def fit(self, gen, steps_per_epoch, epochs, ensemble_index):
        if self.ensemble_size > 1:
            model = self.world_model[ensemble_index]
            opt = self._opt[ensemble_index]
        else:
            model = self.world_model
            opt = self._opt
        for step in range(steps_per_epoch * epochs):
            obs, actions, next_obs = next(gen)
            world_model_pred = model(obs, actions)
            world_model_loss = self.beta * model.loss(world_model_pred, next_obs)
            opt.zero_grad()
            world_model_loss.backward()
            opt.step()

    def _generate_batch(self, obs, actions, next_obs):
        """Generate batches of data used to train the ICM."""
        sample_count = len(actions)
        while True:
            number_of_batches = sample_count // self._batch_size
            for batch_index in range(number_of_batches):
                from_index = batch_index * self._batch_size
                to_index = (batch_index + 1) * self._batch_size
                yield (obs[from_index:to_index],
                        actions[from_index:to_index],
                        next_obs[from_index:to_index])
                
    def checkpoint_world_model(self, path):
        self.curiosity_module.checkpoint_world_model(path)