from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

from zfa_rl_agent.core.utils.logger import logger
import wandb
from wandb.integration.sb3 import WandbCallback

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
            project="zfa_IDM",
            dir=config.log_subdir,
            name=config.run_name,
            config=OmegaConf.to_object(config),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # True = save videos, but doesn't work
            save_code=False,
        )
    
    logger.info("Creating Model")
    model = instantiate(config.agent)
    print("Model Architecture: \n", model.policy)

    logger.info("Making Callbacks")
    eval_callback = instantiate(config.eval_callback)
    checkpoint_callback = instantiate(config.checkpoint_callback)
    wandb_callback = WandbCallback(verbose=2)
    callbacks = [eval_callback, checkpoint_callback, wandb_callback]
    callbacks = [checkpoint_callback, wandb_callback]
    logger.info("Beginning Training")
    model.learn(total_timesteps=config.total_timesteps, 
                callback=callbacks,
                tb_log_name=config.name,
                progress_bar = True)
    model.env.close()
    print(model.env._step_count)
    logger.info("Training Complete")
    run.finish()

if __name__ == "__main__":
    train()