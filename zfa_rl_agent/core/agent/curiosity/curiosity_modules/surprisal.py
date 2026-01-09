import copy
import torch.nn as nn
import torch
import torch.optim as optim


class Surprisal:
    def __init__(self,
                world_model,
                ):
        self.world_model = world_model
        self.name = "surprisal"
        self.feature_size = world_model.input_size
        self.action_size = world_model.action_size
        self.opt = world_model.opt
        self.loss_fn = nn.MSELoss(reduction='none')
        self.reward_calls = 0
    def reward(self, obs, action, obs_next):
        self.reward_calls += 1
        with torch.no_grad():
            world_model_pred = self.world_model(obs, action)
            world_model_loss = self.loss_fn(world_model_pred, obs_next)
            #reward = -torch.log(torch.maximum(world_model_loss, torch.zeros_like(world_model_loss) + 1e-5)).sum(dim=-1)
        return world_model_loss.mean(dim=-1)
    def update(self, obs=None, actions=None, obs_next=None):
        pass
    def checkpoint_world_model(self, path):
        pass
    def reset(self):
        pass
