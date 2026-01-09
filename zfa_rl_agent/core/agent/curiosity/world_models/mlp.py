import torch
from torch import nn
import torch.optim as optim
import numpy as np

class MLPWorldModel(nn.Module):
    def __init__(self, feature_size, action_size, net_arch=[512, 512], lr=1e-3, device='cuda'):
        super(MLPWorldModel, self).__init__()
        self.input_size = feature_size
        self.action_size = action_size
        self.output_size = feature_size
        self.device = device

        layers = []
        dim = self.input_size + self.action_size
        for size in net_arch:
            layers.append(nn.Linear(dim, size))
            layers.append(nn.ReLU())
            dim = size
        layers.append(nn.Linear(dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.mlp.to(self.device)
        self.opt = optim.Adam(self.mlp.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, obs, actions):
        obs = self.to_tensor(obs)
        actions = self.to_tensor(actions)
        obs_and_actions = torch.cat([obs, actions], dim=-1)
        return self.mlp(obs_and_actions)
    
    def loss(self, obs_pred, obs_next):
        obs_pred = self.to_tensor(obs_pred)
        obs_next = self.to_tensor(obs_next)
        return nn.functional.mse_loss(obs_pred, obs_next)
    
    def reset(self):
        self.mlp.apply(self._weight_init)
        self.opt = optim.Adam(self.mlp.parameters(), lr=1e-3)

    def _weight_init(self, m):
        if isinstance(m, nn.Linear):
            #nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, tuple):
            return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:
            raise Exception(f"Unrecognized type of observation {type(x)}")
