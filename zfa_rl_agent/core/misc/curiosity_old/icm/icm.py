"""ICM and some related networks to use ICM"""

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from zfa_rl_agent.core.agent.extractors.feature_net import MultiModalFeatureExtractor

class ICM(nn.Module):
    def __init__(self, feature_dim = 2*64*64 + 2 * 64, action_shape = 5, **kwargs):
        super(ICM, self).__init__()
        self.inverse_model = InverseNet(feature_dim, action_shape)
        self.forward_model = ForwardNet(feature_dim, action_shape)
            
        self.forward_loss = nn.MSELoss(reduction='none')
        self.inverse_loss = nn.CrossEntropyLoss(reduction='none')

        self.beta = kwargs.get('beta', 0.2)

    def forward(self, state_t, action, state_tp1):
        state_t = self._to_tensor(state_t)
        state_tp1 = self._to_tensor(state_tp1)
        action = self._to_tensor(action)

        state_tp1_hat_pred = self.forward_model(state_t.detach(), action.detach())
        forward_error = self.forward_loss(state_tp1_hat_pred, state_tp1.detach()).mean(dim=-1) #shape? 
        action_pred = self.inverse_model(state_t.detach(), state_tp1.detach())

        action_dist1 = action.view(-1, self.action_shape[0])
        action_dist2 = action_pred.view(-1, self.action_shape[0])

        inverse_error = self.inverse_loss(action_dist1, action_dist2.argmax(dim=-1)).view(action.shape[0], action.shape[1])
        return forward_error, inverse_error
    
    def reward(self, state_t, action, state_tp1):
        state_t = self._to_tensor(state_t)
        state_tp1 = self._to_tensor(state_tp1)
        action = self._to_tensor(action)

        state_tp1_hat_pred = self.forward_model(state_t.detach(), action.detach())
        forward_error = self.forward_loss(state_tp1_hat_pred, state_tp1.detach()).mean(dim=-1) #shape?

        icm_reward = torch.zeros_like(forward_error)

        return forward_error
    
    def loss_fn(self, forward_loss, inverse_loss):
        loss_ = (1-self.beta)*inverse_loss
        loss_ += self.beta*forward_loss
        loss_  = loss_.sum()/loss_.flatten().shape[0]
        return loss_
    
    def _to_tensor(self, obs):
        """
        Moves the observation to the given device.

        :param obs:
        :param device: PyTorch device
        :return: PyTorch tensor of the observation on a desired device.
        """
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, dict):
            return {key: torch.tensor(obs[key], dtype=torch.float32) for key in self.keys}
        elif isinstance(obs, torch.Tensor):
            return obs
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")
        
class ForwardNet(nn.Module):
    """
    Forward model. Predicts next state-feature encoding 
    from current feature state-feature encoding and action. 
    """
    def __init__(self, feature_dim, action_shape, **kwargs):
        super(ForwardNet, self).__init__()

        hidden_sizes = kwargs.get('forward_net_arch', [256, 256])
        activation_fn = kwargs.get('icm_activation_fn', nn.ReLU)
        layers = []
        input_dim = feature_dim + action_shape
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation_fn())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, feature_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.model(x)
        
class InverseNet(nn.Module):
    """
    Inverse model. Predicts actions from feature vectors.
    """
    def __init__(self, feature_dim, action_shape, **kwargs):
        super(InverseNet, self).__init__()
        hidden_sizes = kwargs.get('inverse_net_arch', [256, 256])
        activation_fn = kwargs.get('icm_activation_fn', nn.ReLU)

        layers = []
        input_dim = feature_dim*2
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation_fn())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, state1, state2):
        x = torch.cat((state1,state2), dim=-1)
        x = self.model(x)
        x = F.softmax(x, dim=-1)
        return x
    
