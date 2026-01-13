import copy
import torch.nn as nn
import torch
import torch.optim as optim
from pathlib import Path

class LearningProgress:
    def __init__(self,
                world_model,
                ):
        self.world_model = world_model
        self.name = "progress"
        self.feature_size = world_model.input_size
        self.action_size = world_model.action_size
        self.loss_fn = nn.MSELoss(reduction='none')
        self.opt = world_model.opt
        self.reward_calls = 0
        self.device = world_model.device

        self.trailing_world_model = self._create_trailing_world_model().to(self.device)

    def reward(self, obs, action, obs_next):
        # assume observations come in feature space coordinates
        self.reward_calls += 1
        with torch.no_grad():
            world_model_pred = self.world_model(obs, action)
            trailing_world_model_pred = self.trailing_world_model(obs, action)
            world_model_loss, trailing_world_model_loss = self.calculate_loss(world_model_pred, trailing_world_model_pred, obs_next)
            progress = trailing_world_model_loss - world_model_loss
            #progress = torch.log(torch.maximum(progress, torch.zeros_like(progress) + 1e-5))
        return progress, world_model_loss, trailing_world_model_loss
    
    def calculate_loss(self, world_model_pred, trailing_world_model_pred, obs_next):
        world_model_loss = self.loss_fn(world_model_pred, obs_next).mean(dim=-1)
        trailing_world_model_loss = self.loss_fn(trailing_world_model_pred, obs_next).mean(dim=-1)
        return world_model_loss, trailing_world_model_loss
    
    def _create_trailing_world_model(self):
        return copy.deepcopy(self.world_model)
    
    def update(self, obs=None, actions=None, obs_next=None):
        pass

    def checkpoint_world_model(self, path):
        # wm_module_path = path + f"/{self.name}/world_models/"
        # trailing_wm_module_path = path + f"/{self.name}/trailing_world_models/"
        # Path(wm_module_path).mkdir(parents=True, exist_ok=True)
        # Path(trailing_wm_module_path).mkdir(parents=True, exist_ok=True)

        torch.save(self.world_model.state_dict(), path)
        torch.save(self.trailing_world_model.state_dict(), path)

    def reset(self):
        self.world_model.reset()
        self.trailing_world_model = self._create_trailing_world_model()


    

class GammaLearningProgress(LearningProgress):
    def __init__(self, world_model, gamma=0.99):
        super().__init__(world_model)
        self.gamma = gamma
    def update(self, obs=None, actions=None, obs_next=None):
        current_wm_state_dict = self.world_model.state_dict()
        gamma_wm_state_dict = self.trailing_world_model.state_dict()
        for key in current_wm_state_dict:
            gamma_wm_state_dict[key] = self.gamma * gamma_wm_state_dict[key] + \
                            (1 - self.gamma) * current_wm_state_dict[key]

        self.trailing_world_model.load_state_dict(gamma_wm_state_dict)

class PersistentProgress(LearningProgress):
    def __init__(self, world_model, persistent_world_model_path, gamma=0.99):
        super().__init__(world_model)
        self.name = "persistence"
        #self.persistent_world_model = self._create_persistent_world_model(persistent_world_model_path).to(self.device)
        self.trailing_world_model = self._create_persistent_world_model(persistent_world_model_path).to(self.device)
        self.world_model.load_state_dict(self.trailing_world_model.state_dict())
        self.gamma = gamma
    def reward(self, obs, action, obs_next):
        self.reward_calls += 1
        with torch.no_grad():
            world_model_pred = self.world_model(obs, action)
            trailing_world_model_pred = self.trailing_world_model(obs, action)
            # persistent_world_model_pred = self.persistent_world_model(obs, action)
            # current_persistence, trailing_persistence = self.calculate_loss(world_model_pred, trailing_world_model_pred, persistent_world_model_pred)
            # current_world_model_loss, trailing_world_model_loss = self.calculate_loss(world_model_pred, trailing_world_model_pred, obs_next)
            
            world_model_loss, trailing_world_model_loss = self.calculate_loss(world_model_pred, trailing_world_model_pred, obs_next)
            persistence = trailing_world_model_loss - world_model_loss
            # persistence = current_persistence - trailing_persistence
            #trailing_world_model_loss, persistent_trailing_world_model_loss = self.calculate_loss(trailing_world_model_pred, persistent_world_model_pred, obs_next)
            #persistence = persistent_trailing_world_model_loss - persistent_world_model_loss
            #print(persistence, persistent_world_model_loss, persistent_trailing_world_model_loss)
        return persistence, world_model_loss, trailing_world_model_loss

    def _create_persistent_world_model(self, path):
        model = copy.deepcopy(self.world_model)
        model.load_state_dict(torch.load(path))
        return model

class ModelMemoryMismatch(LearningProgress):
    def __init__(self, world_model, memory_path, gamma=0.99):
        super().__init__(world_model)
        self.name = "3m_progress"
        self.memory = self._create_memory(memory_path).to(self.device)
        self.gamma = gamma
        self.ewa_bias = None
        self.huber_loss = nn.HuberLoss(delta=1.0, reduction="none")

    @torch.no_grad()
    def reward(self, obs, action, obs_next):
        self.reward_calls += 1
        if self.reward_calls % 1000 == 0:
            self.ewa_bias=None
        wm_pred = self.world_model(obs, action)
        mem_pred = self.memory(obs, action)
        bias = self.loss_fn(wm_pred, mem_pred)
        self._reward_filter(bias)  
        r_int = torch.abs(bias - self.ewa_bias).mean(dim=-1)  # (B,)
        mem_loss, model_loss = self.calculate_loss(mem_pred, wm_pred, obs_next)
        return r_int, self.ewa_bias.mean(dim=-1), mem_loss, model_loss

    def _create_memory(self, path):
        model = copy.deepcopy(self.world_model)
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model

    def _reward_filter(self, bias):
        if self.ewa_bias is None:
            self.ewa_bias = bias
        else:
            self.ewa_bias = self.gamma * self.ewa_bias + (1 - self.gamma) * bias
            
class BeliefProgress(LearningProgress):
    def __init__(self, world_model, memory_path,
                 gamma=0.998,               # slow horizon ≈1/(1-gamma)
                 cycle_horizon=0.9):            # fast tracker  (τ≈10)
        super().__init__(world_model)
        self.name = "belief_progress"
        self.memory = self._create_memory(memory_path).to(self.device)
        for p in self.memory.parameters():  
            p.requires_grad_(False)

        self.gamma = gamma
        self.beta  = cycle_horizon             
        self.m_slow = None                  
        self.m_fast = None
        self.loss_fn = nn.MSELoss(reduction="none")

    @torch.no_grad()
    def reward(self, obs, action, obs_next):
        zero_a = torch.zeros_like(action)
        resid_cur = ( self.world_model(obs, action)
                    - self.world_model(obs, zero_a) )
        resid_mem = ( self.memory(obs, action)
                    - self.memory(obs, zero_a) )

        bias = self.loss_fn(resid_cur, resid_mem).mean(dim=-1, keepdim=True)  

        if self.m_slow is None:
            self.m_slow = bias.clone()
            self.m_fast = bias.clone()

        self.m_fast = self.beta  * self.m_fast + (1 - self.beta)  * bias
        self.m_slow = self.gamma * self.m_slow + (1 - self.gamma) * bias

        r_int = torch.abs(self.m_fast - self.m_slow).squeeze(-1)    # (B,)
        mem_loss, model_loss = self.calculate_loss(
                                   self.memory(obs, action),
                                   self.world_model(obs, action),
                                   obs_next)

        return r_int, self.m_slow.squeeze(-1), mem_loss, model_loss

    def _create_memory(self, path):
        model = copy.deepcopy(self.world_model)
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model
        

# Not finished yet. Do not use 
class DeltaLearningProgress(LearningProgress):
    def __init__(self, world_model, params):
        super().__init__(world_model, params)

        self.world_model_save_path = params["world_model_save_path"]
        self.delta = params["delta"]
        self.update_count = 0
    def update(self):
        torch.save(self.world_model.model.state_dict(),
            f'{self.world_model_save_path}/delta_world_model_{self.update_count}.pt')
        
        trailing_step = self.update_count - self.delta if self.update_count >= self.delta else 0
        trailing_wm_path = f'{self.world_model_save_path}/delta_world_model_{trailing_step}.pt'
        self.trailing_world_model.model.load_state_dict(torch.load(trailing_wm_path))
        self.update_count += 1