import copy
import torch.nn as nn
import torch
import torch.optim as optim

class Disagreement:
    def __init__(self, world_model, ensemble_size=3):
        self.name = "disagreement"
        self.ensemble_size = ensemble_size
        self.feature_size = world_model.input_size
        self.action_size = world_model.action_size
        self.world_model = [copy.deepcopy(world_model) for _ in range(ensemble_size)]
        self.opt = [wm.opt for wm in self.world_model]
        self.loss_fn = nn.MSELoss(reduction='none')
        self.reward_calls = 0

    def reward(self, obs, action, obs_next):
        self.reward_calls += 1
        with torch.no_grad():
            world_model_preds = torch.stack([model(obs, action) for model in self.world_model])
            disagreement = torch.var(world_model_preds, dim=0).sum(dim=-1)
        return disagreement
    
    def update(self, obs=None, actions=None, obs_next=None):
        "Update handled in curiosity trainer"
        pass
    def checkpoint_world_model(self, path):
        pass
    def reset(self):
        pass