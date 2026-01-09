import logging
import sys
import os
import itertools
from zfa.model_training.optic_flow_train import train
from zfa.core.default_dirs import OPTIC_FLOW_CHKP_DIR, OPTIC_FLOW_LOGS_DIR, OPTIC_FLOW_MODEL_DIR

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class Args:
    def __init__(self):
        self.save_checkpoint = True
        self.num_epochs = 100
        self.gpus = [0]  # Assuming CUDA setup with 1 GPU
        self.mixed_precision = False
        self.epsilon = 1e-8
        self.clip = 1.0
        self.add_noise = True
        self.batch_size = 128
        self.num_workers = 32
        self.size = 64  # 
        self.lr = 1e-1  # Default value
        self.wdecay = 1e-2  # Default value
        self.name = None
        self.checkpoint_path = None
        self.hyperparam_tuning = False
        self.num_inter_epochs = 100

# ALL (hyperparam tuning)
#lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
wdecay_list =  [1e-4, 1e-3, 1e-2]

# lr = .01
#lr_list = [1e-2]

# lr .1
lr_list = [1e-1]

for lr, wd in itertools.product(lr_list, wdecay_list):
    args = Args()
    args.lr = lr
    args.wdecay = wd

    # Dynamically set name and checkpoint path based on hyperparameters
    args.name = f'optic_flow_{args.size}_lr{args.lr}_wd{args.wdecay}'
    
    # Log the current training configuration
    logging.info(f"Starting training for {args.name}")
    
    # Define checkpoint path
    args.checkpoint_path = os.path.join(OPTIC_FLOW_CHKP_DIR, f'{args.name}_latest_checkpoint.pth')
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    # Train the model and collect results
    final_model_path, train_losses, val_losses = train(args)
    final_val_loss = val_losses[-1] if val_losses else float('inf')
    
    # Log final validation loss
    logging.info(f"Training for {args.name} completed with final validation loss: {final_val_loss}")