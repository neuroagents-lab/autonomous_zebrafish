import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import tqdm
import time
import os 
import logging
import sys
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from zfa.models.optic_flow_architetures import Net_3_layers, Net_2_layers
from zfa.model_training.datasets import optic_flow_fetch_dataloader, optic_flow_fetch_dataloader_synthetic
from zfa.core.default_dirs import OPTIC_FLOW_CHKP_DIR, OPTIC_FLOW_LOGS_DIR, OPTIC_FLOW_MODEL_DIR
from zfa.model_training.optic_flow_evaluate import validate_model

MAX_FLOW = 400.0

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def sequence_loss(flow_pred, flow_gt, valid = None, max_flow=MAX_FLOW):
    """ Adding masking and computing L1 loss """
    # Exclude invalid pixels and extremely large displacements
    # Adding  masking to focus on more reasonable or expected motion patterns
    # Especially if such extreme values are likely to be noise
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

def fetch_optimizer(args, model, train_loader):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.num_epochs * len(train_loader),
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train_one_epoch(epoch, train_loader, model, optimizer, scheduler, scaler, device, args):
    """Train the model for one epoch."""
    model.train()  # Set model to training mode
    running_loss = 0.0
    start_epoch_time = time.time()  # Start timing for this epoch    
    start = time.time()
    for i_batch, data_blob in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, file=sys.stdout)):

        optimizer.zero_grad()
        image1, image2, flow = [x.to(device, non_blocking=True) for x in data_blob]

        image1 = image1.float()
        image2 = image2.float()
        flow = flow.float()
        # Optional: Adding noise
        if args.add_noise:
            stdv = np.random.uniform(0.0, 5.0)
            image1 = (image1 + stdv * torch.randn(*image1.shape).to(device)).clamp(0.0, 255.0)
            image2 = (image2 + stdv * torch.randn(*image2.shape).to(device)).clamp(0.0, 255.0)


        with torch.cuda.amp.autocast(enabled=args.mixed_precision): # Set to False
            flow_prediction = model([image1, image2])
            loss = sequence_loss(flow_prediction, flow)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

    avg_epoch_loss = running_loss / len(train_loader)

    # Calculate and display epoch time
    epoch_time = time.time() - start_epoch_time
    logging.info(f"Epoch [{epoch}] completed in {epoch_time:.2f} seconds")

    return avg_epoch_loss, epoch_time


def train(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Net_3_layers(input_size=args.size)
    model = model.float()
    model = model.to(device)

    dataloaders = optic_flow_fetch_dataloader(args)
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val')

    optimizer, scheduler = fetch_optimizer(args, model, train_loader)

    scaler = GradScaler(enabled=args.mixed_precision)

    early_stopper = EarlyStopper(patience=3, min_delta=.01)

    epoch = 1
    train_losses = []
    val_losses = []
    log_dir = os.path.join(OPTIC_FLOW_LOGS_DIR, args.name)

    # Load checkpoint if available
    if args.checkpoint_path is not None:
        if os.path.isfile(args.checkpoint_path):
            logging.info(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path)

            epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint.get('train_losses', train_losses)
            val_losses = checkpoint.get('val_losses', val_losses)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log_dir = checkpoint.get('log_dir')

            logging.info(f"Checkpoint loaded. Starting from epoch {epoch}")
        else:
            logging.info(f"Checkpoint file {args.checkpoint_path} does not exist. Continuing without loading.")

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # CHECKING FOR NAN VALUES
    #torch.autograd.set_detect_anomaly(True)
    # Training loop
    for epoch in range(epoch, args.num_epochs + 1):
        logging.info(f"\nStarting epoch {epoch}/{args.num_epochs}")
        # Train for one epoch
        avg_train_loss, epoch_time = train_one_epoch(epoch, train_loader, model, optimizer, scheduler, scaler, device, args)
        logging.info(f"Epoch [{epoch}] Average Training Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        if val_loader is not None:
            avg_epe_val_loss = validate_model(model, val_loader, device)
            logging.info(f"Epoch [{epoch}] EPE Validation Loss: {avg_epe_val_loss:.4f}")                
            val_losses.append(avg_epe_val_loss)
            writer.add_scalar('EPE Loss/val', avg_epe_val_loss, epoch)        

        # Save checkpoint
        if args.save_checkpoint:
            checkpoint_path = f'{OPTIC_FLOW_CHKP_DIR}epoch_{epoch}_{args.name}.pth'
            latest_checkpoint_path = args.checkpoint_path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'log_dir': log_dir,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")

            # Update the symlink for the latest checkpoint
            if os.path.exists(latest_checkpoint_path):
                os.remove(latest_checkpoint_path)
            os.symlink(checkpoint_path, latest_checkpoint_path)
            logging.info(f"Updated latest checkpoint symlink to {latest_checkpoint_path}")

        # Hyperparameter tuning or stop training
        if (args.hyperparam_tuning and epoch == 5) or (args.num_inter_epochs == epoch):
            break
        
        # Early stopping check
        if early_stopper.early_stop(avg_epe_val_loss):             
            break
    
    writer.close()
    
    # Calculate and display total training time
    total_training_time = time.time() - start_time
    logging.info(f"Total training time: {total_training_time / 60:.2f} minutes")

    # Save final model
    final_model_path = f'{OPTIC_FLOW_MODEL_DIR}{args.name}_best.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at {final_model_path}")
    return final_model_path, train_losses, val_losses