from pathlib import Path
import os, glob
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

from zfa_rl_agent.core.utils.logger import logger
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

class EnvironmentLoggerCallback(BaseCallback):
    """
    Callback that logs which environment is currently being used.
    """
    def __init__(self, env_name, verbose=0):
        super(EnvironmentLoggerCallback, self).__init__(verbose)
        self.env_name = env_name
        
    def _on_step(self) -> bool:
        # Log environment type to tensorboard
        self.logger.record("environment/type", hash(self.env_name) % 2)  # 0 for swim, 1 for drift
        self.logger.record("environment/name", self.env_name)
        return True
    
def get_latest_checkpoint(checkpoint_dir):
    """
    Get the most recently modified checkpoint file in a directory.
    
    Args:
        checkpoint_dir (str): Path to the checkpoint directory
    
    Returns:
        str: Path to the most recent checkpoint file, or None if no files found
    """
    # Ensure directory exists
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory {checkpoint_dir} doesn't exist")
        return None
    
    # Find all zip files in the directory
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Sort by modification time (most recent last)
    checkpoint_files.sort(key=os.path.getmtime)
    
    # Return the most recent file
    latest_checkpoint = checkpoint_files[-1]
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config):
    logger.info("Creating Directories")
    Path(f"{config.log_subdir}").mkdir(parents=True, exist_ok=True)
    Path(config.tb_log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.monitor_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(config.best_model_path).mkdir(parents=True, exist_ok=True)
    Path(config.eval_path).mkdir(parents=True, exist_ok=True)
    logger.info("Saving Config")
    OmegaConf.save(config, f"{config.log_subdir}/config.yaml")

    logger.info(f"Using {config.env_name} environment")

    run = wandb.init(
            project="zfa_multi_test",
            dir=config.log_subdir,
            name=config.run_name,
            config=OmegaConf.to_object(config),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # True = save videos, but doesn't work
            save_code=False,
        )
    
    logger.info("Creating Environments")
    swim_env = instantiate(config.swim_curiosity_environment)
    drift_env = instantiate(config.drift_curiosity_environment)

    def get_model(env, checkpoint_path=None):
        if checkpoint_path:
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            return instantiate(config.load_agent, path=checkpoint_path, env=env)
        else:
            logger.info("Creating new model")
            return instantiate(config.initialize_agent, _partial_=True)(env=env)
            
    logger.info("Making Callbacks")
    #eval_callback = instantiate(config.eval_callback)
    checkpoint_callback = instantiate(config.checkpoint_callback)
    wandb_callback = WandbCallback(verbose=2)
    callbacks = [checkpoint_callback, wandb_callback]

    # Calculate number of switches based on total timesteps and switch rate
    switch_rate = config.switch_rate  
    total_timesteps = config.total_timesteps
    num_switches = total_timesteps // switch_rate
    
    logger.info(f"Will switch environments {num_switches} times during training")
    
    # Keep track of total trained steps
    total_steps_done = 0
    reset_num_timesteps = True  # Only reset for the first phase
    
    # Alternate between environments
    current_checkpoint = get_latest_checkpoint(config.checkpoint_path)
    for i in range(num_switches + 1):
        # Determine current environment for this phase
        if i % 2 == 0:
            current_env = swim_env
            env_name = "swim"
        else:
            current_env = drift_env
            env_name = "drift"
        
        # Get or create model with the right environment
        model = get_model(current_env, current_checkpoint)
        if i == 0:
            print("Model Architecture: \n", model.policy)
        
        # Calculate steps for this phase
        steps_this_phase = min(switch_rate, total_timesteps - total_steps_done)
        if steps_this_phase <= 0:
            break
        # Create the environment logger callback for this phase
        env_logger = EnvironmentLoggerCallback(env_name)
        
        # Add it to the callbacks for this phase
        phase_callbacks = callbacks + [env_logger]
        logger.info(f"Training Phase {i+1}: {env_name} environment for {steps_this_phase} steps")
        model.learn(
            total_timesteps=steps_this_phase,
            callback=phase_callbacks,
            tb_log_name=config.name,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )
        
        # Save checkpoint after this phase
        phase_path = os.path.join(config.checkpoint_path, 
                       f"{config.run_name}_phase{i+1}_{env_name}_{total_steps_done+steps_this_phase}_steps.zip")
        model.save(phase_path)
        logger.info(f"Phase {i+1} complete. Model saved to {phase_path}")
        
        # Update checkpoint for next phase
        current_checkpoint = phase_path
        
        # Update total steps and reset flag
        total_steps_done += steps_this_phase
        reset_num_timesteps = False  # Don't reset for subsequent phases
        
        # Close the environment to free resources
        model.env.close()
    logger.info(f"Training Complete. Total steps: {total_steps_done}")
    swim_env.close()
    drift_env.close()
    run.finish()

if __name__ == "__main__":
    train()