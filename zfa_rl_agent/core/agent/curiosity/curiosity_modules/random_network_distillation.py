import copy
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

class RandomNetworkDistillation:
    def __init__(self,
                world_model,
                ):
        self.world_model = world_model
        self.name = "rnd"
        self.feature_size = world_model.input_size
        self.action_size = world_model.action_size
        self.reward_calls = 0
        self.device = world_model.device  # Get device from world_model

        self.target_net = RNDNetwork(
            observation_dim=self.feature_size, # should be world.model.obs_size
            output_dim=512,
            hidden_layer_arch=[64, 64], 
            device=self.device)

        self.predictor_net = RNDNetwork(
            observation_dim=self.feature_size,
            output_dim=512,
            hidden_layer_arch=[64, 64],
            device=self.device)

        self.opt = torch.optim.Adam(self.predictor_net.parameters(), lr=1e-3)

        # for reward normalization
        self.reward_moments = RunningMoments()
        self.reward_filter = RewardForwardFilter(gamma=0.99) # see paper for gamma

        # for observation normalization
        self.obs_moments = RunningMoments()

        self.normalize_rewards = False
        self.normalize_obs = False

    def reward(self, obs, action=None, obs_next=None):
        # assume observations come in feature space coordinates
        self.reward_calls += 1
        rmse = self._prediction_rmse(obs)
        with torch.no_grad():
            if self.normalize_rewards:
                # update reward normalization params
                moving_average_rewards = torch.tensor([self.reward_filter.update(r) for r in rmse])
                self.reward_moments.update(moving_average_rewards)
                # normalize reward
                reward = rmse / self.reward_moments.get_stdev()
            else:
                reward = rmse
        return reward 
    
    # TODO: "implement" Update the predictor network. 
    def update(self, obs, actions=None, obs_next=None): 
        assert obs is not None, "RND requires observations to update the predictor network."
        obs = self._to_tensor(obs)
        loss = nn.functional.mse_loss(self.target_net(obs), self.predictor_net(obs))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    
    def _prediction_rmse(self, obs):
        if self.normalize_obs:
            obs = self._normalize_obs(obs)

        with torch.no_grad():
            targets = self.target_net(obs)

        predictions = self.predictor_net(obs)
        square_diff = torch.square(targets - predictions)
        rmse = torch.sqrt(torch.mean(square_diff, dim=1)) # TODO: should this be mean?

        return rmse
    
    def _to_tensor(self, obs):
        """
        Moves the observation to the device.

        :param obs: Observation in various formats
        :return: PyTorch tensor of the observation on the device.
        """
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif isinstance(obs, list):
            return torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif isinstance(obs, dict):
            return {key: torch.tensor(obs[key], dtype=torch.float32, device=self.device) for key in self.keys}
        elif isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")
        
    def checkpoint_world_model(self, path):
        pass
        #raise NotImplementedError(f"Checkpointing the {self.name} model is not yet implemented.")

    def reset(self):
        pass



class RNDNetwork(nn.Module):
    def __init__(self,
                 observation_dim: int,
                 output_dim: int,
                 hidden_layer_arch: list[int] = [32, 32],
                 device: str = 'cuda'):

        super(RNDNetwork, self).__init__()
        self.device = device
        self.num_hidden_layers = len(hidden_layer_arch)
        self.dense1 = torch.nn.Linear(observation_dim, hidden_layer_arch[0])

        if self.num_hidden_layers == 1:
            self.dense2 = torch.nn.Linear(hidden_layer_arch[0], output_dim)
            self.dense3 = None
            self.dense4 = None

        elif self.num_hidden_layers == 2:
            self.dense2 = torch.nn.Linear(hidden_layer_arch[0], hidden_layer_arch[1])
            self.dense3 = torch.nn.Linear(hidden_layer_arch[1], output_dim)
            self.dense4 = None

        elif self.num_hidden_layers == 3:
            self.dense2 = torch.nn.Linear(hidden_layer_arch[0], hidden_layer_arch[1])
            self.dense3 = torch.nn.Linear(hidden_layer_arch[1], hidden_layer_arch[2])
            self.dense4 = torch.nn.Linear(hidden_layer_arch[2], output_dim)

        else:
            raise ValueError('Only 1, 2 or 3 hidden layer architecture is supported for RND networks')
        
        # Move the network to the specified device
        self.to(self.device)

    def forward(self, x):
        # Make sure input is on the correct device
        x = x.to(self.device)
        
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)
        x = self.dense2(x)

        if self.num_hidden_layers >= 2:
            x = torch.nn.functional.relu(x)
            x = self.dense3(x)

        if self.num_hidden_layers >= 3:
            x = torch.nn.functional.relu(x)
            x = self.dense4(x)

        return x
    

class RunningMoments(object):
    # Based on the code associated with Burda et al.'s 2018 paper: 'Exploration by Random Network Distillation'

    # Former name: RunningMeanStd (removed MPI elements)
    # The code referenced https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float64, requires_grad=False)
        self.var = torch.ones(shape, dtype=torch.float64, requires_grad=False)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=0)
            batch_std = torch.std(x, dim=0, unbiased=False)
            batch_count = x.shape[0]
            batch_var = torch.square(batch_std)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def get_mean(self):
        return self.mean

    def get_stdev(self):
        return torch.sqrt(self.var)


class RewardForwardFilter(object):
    # Originally from the code associated with Burda et al.'s 2018 paper: 'Exploration by Random Network Distillation'
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems