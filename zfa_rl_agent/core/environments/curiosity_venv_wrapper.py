from zfa_rl_agent.core.agent.curiosity.wrapper import CuriosityWrapper
from zfa_rl_agent.core.agent.curiosity.world_models import MLPWorldModel, LSTMWorldModel
from zfa_rl_agent.core.agent.curiosity.curiosity_modules import (
    Disagreement, 
    GammaLearningProgress, 
    Surprisal, 
    ICM, 
    RandomNetworkDistillation,
    PersistentProgress,
    TaskProgress,
    ModelMemoryMismatch, 
    BeliefProgress
    )
from zfa_rl_agent.core.agent.extractors.torch_layers import CombinedExtractor
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.vec_env import VecNormalize

def curiosity_venv(venv, 
                   feature_size, 
                   action_size, 
                   world_model_class = 'mlp',
                   world_model_kwargs: Optional[dict[str, Any]] = {},
                   curiosity_env_kwargs: Optional[dict[str, Any]] = {},
                   persistent_world_model_path: Optional[str] = None,
                   alpha: Optional[float] = 0.9,
                   gamma: Optional[float] = 0.99,
                   cycle_horizon: Optional[int] = 200,
                   ):
     
     if world_model_class == 'mlp':
          world_model = lambda: MLPWorldModel(feature_size, action_size, **world_model_kwargs)
     elif world_model_class == 'lstm':
          world_model = lambda: LSTMWorldModel(feature_size, action_size, **world_model_kwargs)
     
     module_dict = {'rnd': RandomNetworkDistillation,
                    'progress': GammaLearningProgress,
                    'surprisal': Surprisal,
                    'icm': ICM,
                    'disagreement': Disagreement,
                    'task_progress': TaskProgress,
                    'persistence': PersistentProgress,
                    '3m_progress': ModelMemoryMismatch, 
                    'belief_progress': BeliefProgress,
                    }

     hardset = curiosity_env_kwargs['reward_type']
     if hardset == 'progress':
          args = dict(world_model=world_model(), gamma=gamma)
     elif hardset == '3m_progress':
          args = dict(world_model=world_model(), memory_path=persistent_world_model_path, gamma=alpha)
     elif hardset == 'persistence':
          args = dict(world_model=world_model(), persistent_world_model_path=persistent_world_model_path, gamma=gamma)
     elif hardset == 'belief_progress':
          args = dict(world_model=world_model(), memory_path=persistent_world_model_path, cycle_horizon=cycle_horizon, gamma=alpha)
     elif hardset == 'task_progress':
          args = dict(alpha=alpha)
     else: 
          args = dict()
     module = module_dict[hardset](**args)
     
     # rnd = RandomNetworkDistillation(world_model())
     # progress = GammaLearningProgress(world_model(), gamma)
     # surprisal = Surprisal(world_model())
     # icm = ICM(world_model())
     # disagreement = Disagreement(world_model())
     # #task_progress = TaskProgress(alpha=alpha)
     

     # if persistent_world_model_path is not None:
     #      #persistence = PersistentProgress(world_model(), persistent_world_model_path)
     #      #module_list = [rnd, progress, surprisal, icm, disagreement, task_progress, persistence]
     #      mmm_progress = ModelMemoryMismatch(world_model(), persistent_world_model_path, alpha)
     #      module_list = [rnd, progress, surprisal, icm, disagreement, mmm_progress]
     # else:
     #      module_list = [rnd, progress, surprisal, icm, disagreement]

     module_list = [module]
     curiosity_env = CuriosityWrapper(venv, module_list, **curiosity_env_kwargs)
     #curiosity_env = VecNormalize(curiosity_env)
     return curiosity_env
