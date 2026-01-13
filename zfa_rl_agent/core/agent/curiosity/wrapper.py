from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces
from zfa_rl_agent.core.agent.curiosity import CuriosityTrainer
import numpy as np
from zfa_rl_agent.core.utils.stats import MovingAverage, RewardNormalizer

class CuriosityWrapper(VecEnvWrapper):
  """SB3 Vectorized environment wrapper for intrinsic reward."""

  def __init__(self,
                vec_env,
                curiosity_module_list,
                reward_type='progress',
                scale_task_reward=1.0,
                scale_surrogate_reward=1.0,
                scale_action_penalty=1.0,
                action_history=200, # 6 second window
                passive_threshold=0.3,
                exploration_reward_min_step=1000,
                checkpoint_path=None,
                checkpoint_freq=None,
                trainer_kwargs={}):
        assert isinstance(vec_env.observation_space, (spaces.Box, spaces.Dict)), (
            "Frame Stacking only works with gym.spaces.Box and gym.spaces.Dict observation spaces"
        )
        super().__init__(vec_env, vec_env.observation_space)

        self.trainer_kwargs = trainer_kwargs
        self.curiosity_module_list = curiosity_module_list
        self._reward_type = reward_type
        self._scale_task_reward = scale_task_reward
        self._scale_surrogate_reward = scale_surrogate_reward
        self._exploration_reward_min_step = exploration_reward_min_step
        self._step_reward_checkpoint_path = checkpoint_path
        self._step_reward_checkpoint_freq = checkpoint_freq

        self._step_count = 0
        self._step_rewards = []

        # Initialize observers and module names
        self._observers = []
        self._add_observers()
        assert self._observers, "At least one observer should be added to the wrapper."
        self.module_names = [module.name for module in self.curiosity_module_list]

        # Use dictionaries to store per-module statistics
        self.stats_reward = {name: MovingAverage(capacity=self.venv.num_envs)
                             for name in self.module_names}
        self.episode_reward = {name: [0.0] * self.venv.num_envs
                               for name in self.module_names}

        self._stats_task_reward = MovingAverage(capacity=self.venv.num_envs)
        self._stats_wm_loss = MovingAverage(capacity=self.venv.num_envs)
        self._stats_trailing_wm_loss = MovingAverage(capacity=self.venv.num_envs)
        self._stats_persistence_wm_loss = MovingAverage(capacity=self.venv.num_envs)
        self._stats_persistence_trailing_wm_loss = MovingAverage(capacity=self.venv.num_envs)
        self._stats_persistence = MovingAverage(capacity=self.venv.num_envs)
        self._stats_trailing_persistence = MovingAverage(capacity=self.venv.num_envs)
        
        # 3M progress
        self._stats_memory_loss = MovingAverage(capacity=self.venv.num_envs)
        self._stats_model_loss = MovingAverage(capacity=self.venv.num_envs)
        self._stats_ewa_reward = MovingAverage(capacity=self.venv.num_envs)

        # Action stats
        self.action_history = action_history
        self._scale_action_penalty = scale_action_penalty
        self._stats_action_torques = MovingAverage(capacity=1000) # n_steps

        self._episode_task_reward = [0.0] * self.venv.num_envs
        self._episode_wm_loss = [0.0] * self.venv.num_envs
        self._episode_trailing_wm_loss = [0.0] * self.venv.num_envs
        self._episode_persistence_wm_loss = [0.0] * self.venv.num_envs
        self._episode_persistence_trailing_wm_loss = [0.0] * self.venv.num_envs
        self._episode_persistence = [0.0] * self.venv.num_envs
        self._episode_trailing_persistence = [0.0] * self.venv.num_envs

        # 3M progress
        self._episode_memory_loss = [0.0] * self.venv.num_envs
        self._episode_model_loss = [0.0] * self.venv.num_envs
        self._episode_ewa_reward = [0.0] * self.venv.num_envs

        # Passivity stats
        self._passive_threshold = passive_threshold
        self._episode_passive_steps = [0] * self.venv.num_envs
        self._episode_total_steps = [0] * self.venv.num_envs
        self._episode_state_switches = [0] * self.venv.num_envs
        self._episode_last_state = [None] * self.venv.num_envs
        self._stats_passive_fraction = MovingAverage(capacity=self.venv.num_envs)
        self._stats_state_switches = MovingAverage(capacity=self.venv.num_envs)

        # per-step dynamics
        self._reward_dynamics = MovingAverage(capacity=1000)
        self._penalty_dynamics = MovingAverage(capacity=1000)
        self._memory_dynamics = MovingAverage(capacity=1000)
        self._model_dynamics = MovingAverage(capacity=1000)

        # For intrinsic reward normalization
        self._reward_normalizer = RewardNormalizer(num_envs=self.venv.num_envs, epsilon=1e-8)

  def _add_observers(self):
      for module in self.curiosity_module_list:
          if module.name != 'task_progress':
              self._observers.append(CuriosityTrainer(module, **self.trainer_kwargs))

  def _compute_intrinsic_reward(self, obs, actions, next_obs, rewards, dones, infos):
     
      # Update observers with the new step information.
      for observer in self._observers:
          observer.on_new_observation(obs, actions, next_obs, dones, infos)

      # Compute bonus rewards for each curiosity module.
      curiosity_rewards = {}
      for module in self.curiosity_module_list:
        if module.name == 'progress':
            feature_rewards, wm_loss, trailing_wm_loss = module.reward(obs, actions, next_obs)
            progress_wm_loss = self._ensure_numpy(wm_loss)
            progress_trailing_wm_loss = self._ensure_numpy(trailing_wm_loss)
            self._model_dynamics.add(progress_wm_loss)
            self._memory_dynamics.add(progress_trailing_wm_loss)

        elif module.name == 'persistence':
            feature_rewards, wm_loss, trailing_wm_loss = module.reward(obs, actions, next_obs)
            #trailing_persistence = self._ensure_numpy(trailing_persistence)
            persistence_wm_loss = self._ensure_numpy(wm_loss)
            persistence_trailing_wm_loss = self._ensure_numpy(trailing_wm_loss)
            self._model_dynamics.add(persistence_wm_loss)
            self._memory_dynamics.add(persistence_trailing_wm_loss)

        elif module.name == 'task_progress':
            feature_rewards = module.reward(rewards, dones)
            
        elif module.name == '3m_progress' or module.name == 'belief_progress':
            feature_rewards, ewa_reward, memory_loss, model_loss = module.reward(obs, actions, next_obs)
            memory_loss = self._ensure_numpy(memory_loss)
            model_loss = self._ensure_numpy(model_loss)
            ewa_reward = self._ensure_numpy(ewa_reward)
            self._model_dynamics.add(model_loss)
            self._memory_dynamics.add(memory_loss)

        else:
            feature_rewards = module.reward(obs, actions, next_obs)

        b_rewards = np.array([self._ensure_numpy(feature_rewards[k]) for k in range(self.venv.num_envs)])
        bonus_rewards = np.array([0.0 if d else s for s, d in zip(b_rewards, dones)])
        curiosity_rewards[module.name] = bonus_rewards

      if self._reward_type not in self.module_names:
          raise ValueError(f"Reward type should be one of {self.module_names} but got {self._reward_type}")

      # Pull designated reward type
      bonus_rewards = curiosity_rewards[self._reward_type]
      surrogate_scale = self._scale_surrogate_reward if self._step_count >= self._exploration_reward_min_step else 0.0

      # Normalize bonus reward. Normalization not reflected in stat reporting. 
      std = self._reward_normalizer.update_and_get_std(bonus_rewards)
      normalized_bonus = bonus_rewards / std

      # Update action torque stats.
      self._stats_action_torques.add(self._ensure_numpy(actions))
      last_N_action_std = self._stats_action_torques.std_last_n(self.action_history)
      total_action_std = last_N_action_std.sum(axis=1) # sum over joints
      #last_N_action_mean = self._stats_action_torques.mean_last_n(self.action_history)
      #total_action_mean = last_N_action_mean.sum(axis=1) # sum over joints

      current_action_norm = np.linalg.norm(self._ensure_numpy(actions), axis=1)

      # Store reward dynamics.
      self._reward_dynamics.add(self._ensure_numpy(normalized_bonus))
      self._penalty_dynamics.add(self._ensure_numpy(current_action_norm))

      # Combine task reward, intrinsic reward, and action penalty.
      postprocessed_rewards = (
          self._scale_task_reward * rewards +
          surrogate_scale * normalized_bonus -
          self._scale_action_penalty * current_action_norm
      )
      # TODO: vectorize this since moving average arrays are objects. 
      # Update per-environment statistics.
      for i in range(self.venv.num_envs):
          infos[i]['intrinsic_reward'] = normalized_bonus[i]
          self._episode_task_reward[i] += rewards[i]
          if 'progress' in self.module_names:
              self._episode_wm_loss[i] += progress_wm_loss[i]
              self._episode_trailing_wm_loss[i] += progress_trailing_wm_loss[i]
          if 'persistence' in self.module_names:
              self._episode_persistence_wm_loss[i] += persistence_wm_loss[i]
              self._episode_persistence_trailing_wm_loss[i] += persistence_trailing_wm_loss[i]
              #self._episode_persistence[i] += persistence[i]
              #self._episode_trailing_persistence[i] += trailing_persistence[i]    
          if '3m_progress' in self.module_names or 'belief_progress' in self.module_names:
              self._episode_memory_loss[i] += memory_loss[i]
              self._episode_model_loss[i] += model_loss[i]
              self._episode_ewa_reward[i] += ewa_reward[i]
            
          # Update intrinsic rewards per module.
          for name in self.module_names:
              self.episode_reward[name][i] += curiosity_rewards[name][i]
              if dones[i]:
                  self.stats_reward[name].add(self.episode_reward[name][i])
                  self.episode_reward[name][i] = 0.0

          # Track passivity.
          is_passive = int(total_action_std[i] < (self._passive_threshold * last_N_action_std.shape[-1]))
          self._episode_passive_steps[i] += is_passive
          self._episode_total_steps[i] += 1
          if self._episode_last_state[i] is not None:
              self._episode_state_switches[i] += int(is_passive != self._episode_last_state[i])
          self._episode_last_state[i] = is_passive

          # On episode end, update aggregated stats.
          if dones[i]:
              self._stats_task_reward.add(self._episode_task_reward[i])
              self._episode_task_reward[i] = 0.0
              if 'progress' in self.module_names:
                  self._stats_wm_loss.add(self._episode_wm_loss[i])
                  self._stats_trailing_wm_loss.add(self._episode_trailing_wm_loss[i])
                  self._episode_wm_loss[i] = 0.0
                  self._episode_trailing_wm_loss[i] = 0.0
              if 'persistence' in self.module_names:
                  self._stats_persistence_wm_loss.add(self._episode_persistence_wm_loss[i])
                  self._stats_persistence_trailing_wm_loss.add(self._episode_persistence_trailing_wm_loss[i])
                  #self._stats_persistence.add(self._episode_persistence[i])
                  #self._stats_trailing_persistence.add(self._episode_trailing_persistence[i])
                  self._episode_persistence_wm_loss[i] = 0.0
                  self._episode_persistence_trailing_wm_loss[i] = 0.0
                  #self._episode_persistence[i] = 0.0
                  #self._episode_trailing_persistence[i] = 0.0
              if '3m_progress' in self.module_names: 
                  self._stats_memory_loss.add(self._episode_memory_loss[i])
                  self._stats_model_loss.add(self._episode_model_loss[i])
                  self._stats_ewa_reward.add(self._episode_ewa_reward[i])
                  self._episode_ewa_reward[i] = 0.0
                  self._episode_memory_loss[i] = 0.0
                  self._episode_model_loss[i] = 0.0

              self._stats_passive_fraction.add(self._episode_passive_steps[i] / self._episode_total_steps[i])
              self._stats_state_switches.add(self._episode_state_switches[i])
              self._episode_passive_steps[i] = 0
              self._episode_total_steps[i] = 0
              self._episode_state_switches[i] = 0
              self._episode_last_state[i] = None

      return np.array(postprocessed_rewards)

  def checkpoint_world_model(self, path):
      for observer in self._observers:
          observer.checkpoint_world_model(path)

  def step_wait(self):
    """Overrides VecEnvWrapper.step_wait."""
    observations, rewards, dones, infos = self.venv.step_wait()
    self._step_count += 1
    # if self._step_count % 1000 == 0:
    #     bonus_reward_stat = self.stats_reward.get(self._reward_type, None)
    #     print(f'step={self._step_count} avg_task_reward={self._stats_task_reward.mean()} '
    #         f'avg_bonus_reward={bonus_reward_stat.mean() if bonus_reward_stat else "N/A"} '
    #         f'ir_scale={self._scale_surrogate_reward}')
    # self._step_rewards.append(rewards.copy())
    # if self._step_count % self._step_reward_checkpoint_freq == 0 and self._step_reward_checkpoint_path:
    #     np.save(f"{self._step_reward_checkpoint_path}/step_rewards.npy",
    #             np.array(self._step_rewards))
    return observations, rewards, dones, infos

  def reset(self):
      """Overrides VecEnvWrapper.reset."""
      return self.venv.reset()

  def _ensure_numpy(self, data):
      """Convert tensor to numpy array if needed."""
      if hasattr(data, 'detach'):
          return data.detach().cpu().numpy()
      return np.array(data) if not isinstance(data, np.ndarray) else data
    


  