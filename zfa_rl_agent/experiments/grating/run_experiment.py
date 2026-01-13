from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

from zfa_rl_agent.core.utils.logger import logger
import wandb
from wandb.integration.sb3 import WandbCallback

import psutil
import os
import time

def log_process_stats():
    print("=== Process Stats ===")
    print(f"PID: {os.getpid()}, CPU%: {psutil.Process().cpu_percent(interval=1)}")
    for child in psutil.Process().children(recursive=True):
        try:
            print(f"  Child PID: {child.pid} | CPU%: {child.cpu_percent()} | CMD: {child.cmdline()}")
        except psutil.NoSuchProcess:
            pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config):
    logger.info("Creating Directories")
    Path(f"{config.log_subdir}").mkdir(parents=True, exist_ok=True)
    Path(config.tb_log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.monitor_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(config.best_model_path).mkdir(parents=True, exist_ok=True)
    Path(config.eval_path).mkdir(parents=True, exist_ok=True)
    Path(config.wm_checkpoint_path).mkdir(parents=True, exist_ok=True)

    logger.info("Saving Config")
    OmegaConf.save(config, f"{config.log_subdir}/config.yaml")

    logger.info(f"Using {config.env_name} environment")

    # wandb.tensorboard.patch(
    #     root_logdir=config.tb_log_dir,
    #     save=False,  # Don't save TensorBoard events to W&B
    #     pytorch=True,  # Auto-detect PyTorch
    #     )    
    
    run = wandb.init(
                project="zfa_IDM",
                dir=config.log_subdir,
                #root_logdir=config.tb_log_dir,
                #group="idm_training",
                name=config.run_name,
                #tags=["train_metrics"], 
                config=OmegaConf.to_object(config),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # True = save videos, but doesn't work
                save_code=False,
            )
    if config.load_dmc_agent:
        logger.info("Loading Model")
        model = instantiate(config.load_agent)
    else:
        logger.info("Creating Model")
        model = instantiate(config.agent)

    print("Model Architecture: \n", model.policy)

    logger.info("Making Callbacks")
    
    callbacks = []
    if config.checkpointing:
        checkpoint_callback = instantiate(config.checkpoint_callback)
        callbacks.append(checkpoint_callback)
    if config.wm_checkpointing:
        wm_checkpoint_callback = instantiate(config.wm_checkpoint_callback)
        callbacks.append(wm_checkpoint_callback)

    wandb_callback = WandbCallback(verbose=2)
    callbacks.append(wandb_callback)
    logger.info("Beginning Training")

    log_process_stats()
    
    model.learn(total_timesteps=config.total_timesteps, 
                callback=callbacks,
                tb_log_name=config.name)
    model.env.close()
    print(model.env._step_count)
    logger.info("Training Complete")
    run.finish()

if __name__ == "__main__":
    train()