from abc import abstractmethod
import copy

import torch.nn as nn
from ...extractors.feature_net import MultiModalFeatureExtractor
import torch
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces

class LearningProgress:
    def __init__(self,
                world_model,
                ):
        
        # this loss fn determines how progress is calculated. the WM may have a different loss fn
        self.loss_fn = nn.MSELoss(reduction='none')
        self.current_world_model = world_model
        self.trailing_world_model = self._create_trailing_world_model()
        self.reward_calls = 0

    def reward(self, obs, action, obs_next):
        # assume observations come in feature space coordinates
        self.reward_calls += 1
        with torch.no_grad():
            world_model_pred = self.current_world_model(obs, action)
            trailing_world_model_pred = self.trailing_world_model(obs, action)
            world_model_loss, trailing_world_model_loss = self.calculate_loss(world_model_pred, trailing_world_model_pred, obs_next)
            progress = (trailing_world_model_loss - world_model_loss).sum(dim=-1)
        return progress
    
    def calculate_loss(self, world_model_pred, trailing_world_model_pred, obs_next):
        world_model_loss = self.loss_fn(world_model_pred, obs_next)
        trailing_world_model_loss = self.loss_fn(trailing_world_model_pred, obs_next)
        return world_model_loss, trailing_world_model_loss
    
    def _create_trailing_world_model(self):
        return copy.deepcopy(self.current_world_model)
    

class GammaLearningProgress(LearningProgress):
    def __init__(self, world_model, gamma=0.99):
        super().__init__(world_model)
        self.gamma = gamma
    def update(self):
        current_wm_state_dict = self.current_world_model.state_dict()
        gamma_wm_state_dict = self.trailing_world_model.state_dict()
        for key in current_wm_state_dict:
            gamma_wm_state_dict[key] = self.gamma * gamma_wm_state_dict[key] + \
                            (1 - self.gamma) * current_wm_state_dict[key]

        self.trailing_world_model.load_state_dict(gamma_wm_state_dict)

# Not finished yet. Do not use 
class DeltaLearningProgress(LearningProgress):
    def __init__(self, world_model, params):
        super().__init__(world_model, params)

        self.world_model_save_path = params["world_model_save_path"]
        self.delta = params["delta"]
        self.update_count = 0
    def update(self):
        torch.save(self.current_world_model.model.state_dict(),
            f'{self.world_model_save_path}/delta_world_model_{self.update_count}.pt')
        
        trailing_step = self.update_count - self.delta if self.update_count >= self.delta else 0
        trailing_wm_path = f'{self.world_model_save_path}/delta_world_model_{trailing_step}.pt'
        self.trailing_world_model.model.load_state_dict(torch.load(trailing_wm_path))
        self.update_count += 1