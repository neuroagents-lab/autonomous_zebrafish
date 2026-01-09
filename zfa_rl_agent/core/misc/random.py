    def concatenate(self, observations: TensorDict, next_observations: TensorDict) -> TensorDict:
        processed_obs = {}
        proprio_list = []
        for key, obs_t, obs_tp1 in zip(self._observation_space.keys(), observations.values(), next_observations.values()):
            if key == 'pixels':
                processed_obs[key] = [obs_t, obs_tp1]
                #processed_obs[key] = obs_t
            else:
                proprio_list.append(obs_t)
        processed_obs['proprio'] = torch.cat(proprio_list, dim=-1)
        return processed_obs
    
    def concatenate_proprio(self, observations: TensorDict) -> TensorDict:
        processed_obs = {}
        proprio_list = []
        for key, obs in observations.items():
            if key == 'pixels':
                processed_obs[key] = obs
            else:
                proprio_list.append(obs)
        processed_obs['proprio'] = torch.cat(proprio_list, dim=-1)
        return processed_obs
    
    def unstack_obs(self, observations: TensorDict) -> tuple[TensorDict]:
        obs = self._to_tensor(observations)
        obs_t = {}
        obs_tp1 = {}
        for key in self._observation_space.keys():
            if key == 'pixels':
                t_idx = 3
                channels_first = obs[key].shape[-1] == 64
                if obs[key].dim() == 5:
                    o_t = obs[key][:, :, :t_idx, ...]
                    o_tp1 = obs[key][:, :, t_idx:, ...]
                elif obs[key].dim() == 4:
                    o_t = obs[key][:, :t_idx, ...]
                    o_tp1 = obs[key][:, t_idx:, ...]
                elif obs[key].dim() == 3 and not channels_first:
                    o_t = obs[key][..., :t_idx]
                    o_tp1 = obs[key][..., t_idx:]

                else:
                    raise ValueError(f"Invalid dimension for key {key} of shape {obs[key].shape}")
            else:
                t_idx = self._observation_space[key].shape[0] // 2
                o_t = obs[key][..., :t_idx]
                o_tp1 = obs[key][..., t_idx:]
            obs_t[key] = o_t
            obs_tp1[key] = o_tp1
        return obs_t, obs_tp1  
    
    def _to_tensor(self, obs):
        """
        Moves the observation to the given device.

        :param obs:
        :param device: PyTorch device
        :return: PyTorch tensor of the observation on a desired device.
        """
        #print(type(obs), obs,)
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, dict):
            return {key: torch.as_tensor(obs[key], dtype=torch.float32) for key in obs.keys()}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")
    