from __future__ import print_function, division
import sys
import math
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
import importlib.util

# Load the script as a module
script_path = os.path.expanduser('~/zebrafish_agent/scripts/python_scripts/akt-optic-flow-training.py')
spec = importlib.util.spec_from_file_location("akt-optic-flow-training", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Access the classes from the dynamically loaded module
OpticFlowDataset = module.OpticFlowDataset
FlowAugmentor = module.FlowAugmentor
Transform = module.Transform
fetch_dataloader = module.fetch_dataloader

# Load the script as a module
script_path = os.path.expanduser('~/zebrafish_agent/scripts/python_scripts/akt-optic-flow-architectures.py')
spec = importlib.util.spec_from_file_location("akt-optic-flow-architectures", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Access the classes from the dynamically loaded module
Net_3_layers = module.Net_3_layers


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(flow_pred, flow_gt, valid = None, max_flow=MAX_FLOW):
    """ Adding masking and computing L1 loss """
    # Exclude invalid pixels and extremely large displacements
    # Adding  masking to focus on more reaseonable or expected motion patterns
    # Especially if such extreme values are likely to be noise or artifacts in your dataset, as RAFT does
    if valid is None:
        # Create a valid mask of ones
        valid = torch.ones_like(flow_gt[:, 0, :, :])
    mag = torch.sum(flow_gt**2, dim=1).sqrt()  
    valid = (valid >= 0.5) & (mag < max_flow) 

    # Compute L1 loss (absolute difference) with masking for valid pixels
    # Doing L1 over L2 to be robust to outliers
    i_loss = (flow_pred - flow_gt).abs()  
    flow_loss = (valid[:, None] * i_loss).mean() 

    return flow_loss

def count_parameters(model):
    """ Count number of parameters in model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model, dataset_size):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    num_steps = math.ceil(dataset_size / args.batch_size)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, num_steps,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    def __init__(self, model, scheduler, log_dir):
        self.model = model
        self.scheduler = scheduler
        self.current_epoch = 0
        self.running_loss = {}
        self.writer = None
        self.log_dir = log_dir

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[Epoch: {:3d}, LR: {:10.7f}] ".format(self.current_epoch + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # Print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.current_epoch)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        # No longer tracking steps, just metrics per epoch
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.current_epoch)

    def close(self):
        self.writer.close()

    def start_epoch(self):
        """ Call this at the start of each epoch to reset running loss and increment epoch count """
        self.current_epoch += 1
        self.running_loss = {}  # Reset running loss for new epoch


def train(args):
    """ Train model """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your custom model
    model = Net_3_layers(input_size=args.size)
    model = nn.DataParallel(model, device_ids=args.gpus).to(device)
    print("Parameter Count: %d" % count_parameters(model))

    # Optionally load a checkpoint
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path), strict=False)

    model.train()

    # Prepare your data loader
    dataloaders = fetch_dataloader(args)
    train_loader = dataloaders['train']

    dataset_size = len(train_loader.dataset)

    # Initialize optimizer and scheduler
    optimizer, scheduler = fetch_optimizer(args, model, dataset_size)

    logger = Logger(model, scheduler, args.log_dir)

    # Optionally initialize GradScaler for mixed precision
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Training loop over epochs
    for epoch in range(args.num_epochs):  # Use args.num_epochs to control the number of epochs
        print(f"Starting epoch {epoch + 1}/{args.num_epochs}...")

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow = [x.to(device) for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn_like(image1)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn_like(image2)).clamp(0.0, 255.0)

            # Forward pass with optional mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    flow_pred = model([image1, image2])
                    loss = sequence_loss(flow_pred, flow)
            else:
                flow_pred = model([image1, image2])
                loss = sequence_loss(flow_pred, flow)

            # Backward pass and optimization
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            scheduler.step()

            # Logging
            metrics = {'loss': loss.item()}
            logger.push(metrics)

        # Validation and checkpointing at the end of every epoch
        print(f"Epoch {epoch + 1}/{args.num_epochs} completed. Running validation...")

        # Save checkpoint if necessary
        if args.save_checkpoint:
            PATH = f'checkpoints/{epoch + 1}_{args.name}.pth'
            torch.save(model.state_dict(), PATH)

        # Perform validation after each epoch
        results = validate_model(model, dataloaders['val'], device=device)
        logger.write_dict(results)
        model.train()  # Set model back to training mode

    logger.close()
    PATH = f'/home/akirscht/zebrafish_agent/tensorboard/checkpoints/{args.name}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


@torch.no_grad()
def validate_model(model, val_dataset, device, iters=24):
    """
    Perform evaluation on a validation dataset passed as a parameter.
    """
    model.eval()  # Set model to evaluation mode
    epe_list = []

    # Iterate over the validation dataset (now over the DataLoader)
    for i_batch, data_blob in enumerate(val_dataset):
        # Get input images and ground truth flow
        image1, image2, flow_gt = [x.to(device) for x in data_blob]

        

        # Forward pass: get predicted flow
        flow_pr = model([image1, image2])
        
        # Compute End-Point Error (EPE)
        print(f'flow_pr: {flow_pr[0].shape}')
        print(f'flow_gt: {flow_gt.shape}')
        epe = torch.sum((flow_pr[0].cpu() - flow_gt.cpu())**2, dim=0).sqrt()  # EPE per pixel
        epe_list.append(epe.view(-1).numpy())  # Store as numpy for aggregation

    # Calculate mean EPE over the whole dataset
    epe = np.mean(np.concatenate(epe_list))
    print(f"Validation EPE: {epe:.6f}")

    return {'epe': epe}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optic Flow Training Arguments")
    parser.add_argument('--name', type=str, default='optic_flow_model', help='Name for saving the model checkpoint')

    # DataLoader parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training, validation, and test sets')

    # Model input size (image size)
    parser.add_argument('--size', type=int, default=64, help='Size of the images (16, 64)')

    # Dataset paths
    parser.add_argument('--root_dir', type=str, default='/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit/',
                        help='Root directory for the dataset')
    parser.add_argument('--flow_dir', type=str, default='/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit/processed/MarineVideoKit_flows/',
                        help='Directory for the optic flow data')

    # Additional arguments
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizer')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='Weight decay for AdamW optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for optimizer')
    parser.add_argument('--num_steps', type=int, default=None, help='Total number of training steps')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='List of GPU IDs to use for training')
    parser.add_argument('--mixed_precision', action='store_true', help='Flag for using mixed precision training')

    # Split for dataset
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help="Dataset split to use ('train', 'val', 'test')")

    # Model and training options
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a model checkpoint to restore')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save model checkpoints')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')

    # Validation and test set specific parameters
    parser.add_argument('--val_freq', type=int, default=5, help='Validation frequency (every n epochs)')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Test batch size')

    # Parse the arguments
    args = parser.parse_args()

    # Create augmentation params dictionary
    args.aug_params = {
        'crop_size': tuple(args.crop_size),
        'min_scale': args.min_scale,
        'max_scale': args.max_scale,
        'do_flip': args.do_flip,
        'size': args.size
    }

    train(args)