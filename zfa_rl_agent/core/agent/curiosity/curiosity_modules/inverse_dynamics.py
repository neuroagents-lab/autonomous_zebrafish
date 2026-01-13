"""ICM and some related networks to use ICM"""

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from zfa_rl_agent.core.agent.extractors.feature_net import MultiModalFeatureExtractor

class ICM:
    def __init__(self, world_model):
        super(ICM, self).__init__()
        self.world_model = world_model
        self.name = "icm"
        self.feature_size = world_model.input_size
        self.action_size = world_model.action_size
        self.device = world_model.device

        self.inverse_model = InverseNet(self.feature_size, self.action_size).to(self.device)
        #self.inverse_loss_fn = nn.CrossEntropyLoss()
        self.inverse_loss_fn = nn.MSELoss() 


        self.beta = 0.2
        self.opt = world_model.opt
        self.inverse_opt = torch.optim.Adam(self.inverse_model.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss(reduction='none')

    def reward(self, state_t, action, state_tp1):
        state_t = self._to_tensor(state_t)
        state_tp1 = self._to_tensor(state_tp1)
        action = self._to_tensor(action)

        state_tp1_hat_pred = self.world_model(state_t, action)
        reward = self.loss_fn(state_tp1_hat_pred, state_tp1).mean(dim=-1) #shape?
        return reward
    
    def calculate_inverse_loss(self, state_t, action, state_tp1):
        state_t = self._to_tensor(state_t)
        state_tp1 = self._to_tensor(state_tp1)
        action = self._to_tensor(action)

        action_pred = self.inverse_model(state_t, state_tp1)
        action = action.view(-1, self.action_size)  # Shape: (400, 5)
        action_pred = action_pred.view(-1, self.action_size)  # Shape: (400, 5)
        #inverse_loss = self.inverse_loss_fn(action_pred, action.argmax(dim=-1))
        inverse_loss = F.mse_loss(action_pred, action)

        return inverse_loss
    
    # def icm_loss(self, forward_loss, inverse_loss):
    #     loss_ = (1-self.beta)*inverse_loss
    #     loss_ += self.beta*forward_loss
    #     loss_  = loss_.sum()/loss_.flatten().shape[0]
    #     return loss_
    
    def update(self, obs, actions, obs_next):
        assert (obs, actions, obs_next) is not (None, None, None), "(s, a, s') is required for pathak-ICM update"
        loss = (1-self.beta) * self.calculate_inverse_loss(obs, actions, obs_next)
        self.inverse_opt.zero_grad()
        loss.backward()
        self.inverse_opt.step()

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:
            raise TypeError(f"Cannot convert {type(x)} to tensor")

    def checkpoint_world_model(self, path):
        pass

    def reset(self):
        pass

class InverseNet(nn.Module):
    """
    Inverse model. Predicts actions from feature vectors.
    """
    def __init__(self, feature_dim, action_size, hidden_sizes = [64, 64], activation_fn = nn.ReLU):
        super(InverseNet, self).__init__()

        layers = []
        input_dim = feature_dim*2
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation_fn())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, state1, state2):
        x = torch.cat((state1,state2), dim=-1)
        x = self.model(x)
        #x = F.softmax(x, dim=-1)
        return x
    
# # Don't need this standalone. World model replaces this. 
# class ForwardNet(nn.Module):
#     """
#     Forward model. Predicts next state-feature encoding 
#     from current feature state-feature encoding and action. 
#     """
#     def __init__(self, feature_dim, hidden_sizes = [64, 64], activation_fn = nn.ReLU):
#         super(ForwardNet, self).__init__()

#         layers = []
#         input_dim = feature_dim
#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(input_dim, hidden_size))
#             layers.append(activation_fn())
#             input_dim = hidden_size
#         layers.append(nn.Linear(input_dim, feature_dim))
#         self.model = nn.Sequential(*layers)

#     def forward(self, state, action):
#         x = torch.cat((state, action), dim=-1)
#         return self.model(x)