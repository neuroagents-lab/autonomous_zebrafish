import torch
from torch import nn
import torch.optim as optim
import numpy as np

class LSTMWorldModel(nn.Module):
    def __init__(self, feature_size, action_size, hidden_size=512, num_layers=2, lr=1e-3):
        super(LSTMWorldModel, self).__init__()
        self.input_size = feature_size
        self.action_size = action_size
        self.output_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(feature_size + action_size, hidden_size)
        self.activation = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_size, feature_size)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.reset()

    def forward(self, obs, actions, state=None):
        obs = self.to_tensor(obs)
        actions = self.to_tensor(actions)
        combined = torch.cat([obs, actions], dim=-1)
        proj_input = self.activation(self.input_proj(combined))
        
        # Add time dimension 
        if proj_input.dim() == 2:
            proj_input = proj_input.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
        
        if state is None:
            batch_size = proj_input.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=proj_input.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=proj_input.device)
            state = (h0, c0)
        
        lstm_out, state = self.lstm(proj_input, state)
        next_obs_pred = self.output_proj(lstm_out.squeeze(1))
        
        return next_obs_pred
        #return next_obs_pred, state
    
    def loss(self, obs_pred, obs_next):
        obs_pred = self.to_tensor(obs_pred)
        obs_next = self.to_tensor(obs_next)
        return self.loss_fn(obs_pred, obs_next)
    
    def reset(self):
        self.apply(self._weight_init)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)

    def _weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, tuple):
            return torch.tensor(np.array(x), dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise Exception(f"Unrecognized type of observation {type(x)}")
