import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
import math
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torchvision.transforms import ColorJitter
import torch.nn.functional as F
from scipy.ndimage import zoom

MAX_FLOW = 400

class OpticFlowDataset(Dataset):
    def __init__(self, 
                root_dir= '/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit/',
                flow_dir= '/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit/processed/MarineVideoKit_flows/', 
                split='train',
                sparse=False,
                transform = None,
                aug_params=None):


        self.frame_dir = os.path.join(root_dir, split)
        self.flow_dir = os.path.join(flow_dir, split)
        self.sparse = sparse
        self.transform = transform
        self.augmentor = None
        if aug_params:
            self.augmentor = FlowAugmentor(**aug_params)

        # Collect frame pairs and corresponding flow files
        self.frame_pairs = []
        self.flow_files = []
        self.match_files()

    def match_files(self):
        # Collect subdirectories for each region
        self.frame_subdirs = sorted([os.path.join(self.frame_dir, subdir) for subdir in os.listdir(self.frame_dir)])
        self.flow_subdirs = sorted([os.path.join(self.flow_dir, subdir) for subdir in os.listdir(self.flow_dir)])

        for frame_subdir, flow_subdir in zip(self.frame_subdirs, self.flow_subdirs):
            # Get sorted list of frames and flows
            frame_files = sorted([os.path.join(frame_subdir, f) for f in os.listdir(frame_subdir) if f.endswith('.png')])
            flow_files = sorted([os.path.join(flow_subdir, f) for f in os.listdir(flow_subdir) if f.endswith('.flo')])

            # Match consecutive frames and flow files
            for i in range(len(flow_files)):
                if i + 1 < len(frame_files):  # Ensure no index out of bounds
                    self.frame_pairs.append((frame_files[i], frame_files[i + 1]))  # Pair consecutive frames
                    self.flow_files.append(flow_files[i])  # Corresponding flow

    def read_flo_file(self, fn):
        """ Read .flo file in Middlebury format """
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise ValueError('Magic number incorrect. Invalid .flo file')
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            flow = np.reshape(data, (h, w, 2))  # (height, width, 2)
        return flow

    def __len__(self):
        return len(self.flow_files)


    def __getitem__(self, idx):
        frame1_file, frame2_file = self.frame_pairs[idx]
        flow_file = self.flow_files[idx]

        # Load frames and flow
        frame1 = Image.open(frame1_file).convert('RGB')
        frame2 = Image.open(frame2_file).convert('RGB')
        flow = self.read_flo_file(flow_file)

        if self.augmentor:
            frame1, frame2, flow = self.augmentor(frame1, frame2, flow)

        # Apply transformations if any
        if self.transform:
            frame1, frame2, flow = self.transform(frame1, frame2, flow)

        # Convert to torch tensors
        frame1 = torch.from_numpy(np.array(frame1)).permute(2, 0, 1).float()  # [C, H, W]
        frame2 = torch.from_numpy(np.array(frame2)).permute(2, 0, 1).float()  # [C, H, W]
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()  # [2, H, W]

        return frame1, frame2, flow

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, size=16):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.color_aug_prob = 0.2

        self.size = size

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        # Ensure scaling doesn't shrink the image too much
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # Apply scaling to images and flow
        if np.random.rand() < self.spatial_aug_prob:
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y] # Scale vector magnitudes too

        # Perform flipping if applicable
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]


        # Ensure the image is large enough for cropping
        assert self.crop_size[0] < img1.shape[0]
        assert self.crop_size[1] < img1.shape[1]

        crop_ht = self.crop_size[0]
        crop_wd = self.crop_size[1]

        # Sample random top-left corner for cropping
        if img1.shape[0] > crop_ht and img1.shape[1] > crop_wd:
            y0 = np.random.randint(0, img1.shape[0] - crop_ht)
            x0 = np.random.randint(0, img1.shape[1] - crop_wd)

            # Crop the images and flow
            img1 = img1[y0:y0+crop_ht, x0:x0+crop_wd]
            img2 = img2[y0:y0+crop_ht, x0:x0+crop_wd]
            flow = flow[y0:y0+crop_ht, x0:x0+crop_wd]
        else:
            raise ValueError(f"Image size {img1.shape} is too small for crop size {self.crop_size}")

        return img1, img2, flow

    
    def resize_to_fixed(self, img1, img2, flow):
        scale_y = self.size / img1.shape[0]
        scale_x = self.size / img1.shape[1]

        img1 = cv2.resize(img1, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        flow = flow * [scale_x, scale_y] # Scale vector magnitudes too
        

        return img1, img2, flow
    
    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1, img2, flow = self.resize_to_fixed(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class Transform:
    def __init__(self, size):
        self.size = size

    def resize(self, img1, img2, flow):
        # Check if img1 is a PIL image or NumPy array
        if isinstance(img1, Image.Image):
            width, height = img1.size  # PIL image
        elif isinstance(img1, np.ndarray):
            height, width = img1.shape[:2]  # NumPy array
        else:
            raise TypeError(f"Unexpected image type: {type(img1)}")

        # Compute the scaling factors
        scale_y = self.size / height
        scale_x = self.size / width

        # Resize images and flow
        img1_resized = img1.resize((self.size, self.size), Image.BILINEAR) if isinstance(img1, Image.Image) else cv2.resize(img1, (self.size, self.size))
        img2_resized = img2.resize((self.size, self.size), Image.BILINEAR) if isinstance(img2, Image.Image) else cv2.resize(img2, (self.size, self.size))
        flow_resized = zoom(flow, (scale_y, scale_x, 1), order=1)
        flow_resized[..., 0] *= scale_x
        flow_resized[..., 1] *= scale_y

        #  Return all three values
        return np.array(img1_resized), np.array(img2_resized), flow_resized
    
    def __call__(self, img1, img2, flow):
        return self.resize(img1, img2, flow)


def fetch_dataloader(args):
    """
    Function to return DataLoaders for train, val, and test datasets.
    """

    # Input size for resizing
    size = args.size  

    # Define augmentation parameters for training
    aug_params_train = aug_params = {
        
        'crop_size': (256, 448), 
        'min_scale': -0.2,        # Scale down to 80% of the original size
        'max_scale': 0.2,         # Scale up to 120% of the original size
        'do_flip': True, 
        'size': size          
    }
    # Create transforms
    transform = Transform(size=size)

    # Initialize datasets based on the split
    datasets = {
        'train': OpticFlowDataset(
            split='train',
            transform = None,
            aug_params=aug_params_train,
        ),
        'val': OpticFlowDataset(
            split='val',
            transform=transform,
            aug_params=None,  # No augmentation for validation
        ),
        'test': OpticFlowDataset(
            split='test',
            transform=transform,
            aug_params=None,  # No augmentation for test
        )
    }

    # Initialize DataLoaders for each split
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=args.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=args.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=args.batch_size,
            shuffle=False,  # No shuffling for testing
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    }

    return dataloaders


def sequence_loss(flow_pred, flow_gt, valid = None, max_flow=MAX_FLOW):
    """ Adding masking and computing L1 loss """
    # Exclude invalid pixels and extremely large displacements
    # Adding  masking to focus on more reaseonable or expected motion patterns
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

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    def __init__(self, model, scheduler, log_dir):
        self.model = model
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir=log_dir)  # Initialize the writer early
        self.log_dir = log_dir
        self.epoch_loss = {}  # Track metrics for the current epoch
        self.current_epoch = 0  # Track the current epoch
        self.running_loss = {}  # Track intra-epoch metrics (optional)

    def _print_training_status(self, epoch):
        metrics_data = [self.epoch_loss[k] for k in sorted(self.epoch_loss.keys())]
        lr = self.scheduler.get_last_lr()[0]
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in self.epoch_loss.items())
        
        # Print training status for the epoch
        print(f"Epoch [{epoch}] | LR: {lr:.7f} | {metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Log metrics to TensorBoard
        for key, value in self.epoch_loss.items():
            self.writer.add_scalar(f"Loss/{key}_epoch", value, epoch)
        
        # Reset epoch loss after logging
        self.epoch_loss = {}

    def push_batch_loss(self, metrics, batch_index, epoch_index):
        """Log batch-wise metrics (optional for intra-epoch tracking)."""
        for key, value in metrics.items():
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += value

        # Optionally log batch-wise metrics to TensorBoard
        if (batch_index + 1) % 100 == 0:  # Log every 100 batches
            for key, value in self.running_loss.items():
                avg_loss = value / 100
                self.writer.add_scalar(f"Loss/{key}_batch", avg_loss, epoch_index * 100 + batch_index)
            self.running_loss = {}  # Reset running loss
    
    def push_epoch_loss(self, metrics):
        """Accumulate metrics for the epoch."""
        for key, value in metrics.items():
            if key not in self.epoch_loss:
                self.epoch_loss[key] = 0.0
            self.epoch_loss[key] += value

    def write_epoch_results(self, results, epoch):
        """Log validation results at the end of the epoch."""
        for key, value in results.items():
            self.writer.add_scalar(f"Val/{key}", value, epoch)

    def close(self):
        self.writer.close()

def train_one_epoch(epoch, train_loader, model, optimizer, scheduler, scaler, device, logger, args):
    """Train the model for one epoch."""
    model.train()  # Set model to training mode
    running_loss = 0.0

    for i_batch, data_blob in enumerate(train_loader):
        optimizer.zero_grad()
        image1, image2, flow = [x.to(device) for x in data_blob]


        if args.add_noise:
            stdv = np.random.uniform(0.0, 5.0)
            image1 = (image1 + stdv * torch.randn_like(image1)).clamp(0.0, 255.0)
            image2 = (image2 + stdv * torch.randn_like(image2)).clamp(0.0, 255.0)

        # Forward pass
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

        # Step the scheduler
        scheduler.step()

        # Log batch loss (optional)
        logger.push_batch_loss({'loss': loss.item()}, i_batch, epoch)

        running_loss += loss.item()

    avg_epoch_loss = running_loss / len(train_loader)
    logger.push_epoch_loss({'train_loss': avg_epoch_loss})
    return avg_epoch_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Net_3_layers(input_size=args.size)
    model = nn.DataParallel(model, device_ids=args.gpus).to(device)

    # Load checkpoint if available
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Prepare data loaders and optimizer
    dataloaders = fetch_dataloader(args)
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val')
    optimizer, scheduler = fetch_optimizer(args, model)

    # Initialize logger
    logger = Logger(model, scheduler, args.log_dir)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nStarting epoch {epoch}/{args.num_epochs}")

        # Train for one epoch
        avg_train_loss = train_one_epoch(epoch, train_loader, model, optimizer, scheduler, scaler, device, logger, args)
        print(f"Epoch [{epoch}] Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    val_image1, val_image2, val_flow = [x.to(device) for x in val_data]
                    val_flow_pred = model([val_image1, val_image2])
                    val_loss_batch = sequence_loss(val_flow_pred, val_flow)
                    val_loss += val_loss_batch.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.write_epoch_results({'val_loss': avg_val_loss}, epoch)
            print(f"Epoch [{epoch}] Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        if args.save_checkpoint:
            checkpoint_path = f'checkpoints/epoch_{epoch}_{args.name}.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Close logger and save final model
    logger.close()
    final_model_path = f'/home/akirscht/zebrafish_agent/tensorboard/checkpoints/{args.name}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
    return final_model_path


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
        epe = torch.sum((flow_pr.cpu() - flow_gt.cpu())**2, dim=0).sqrt()  # EPE per pixel
        epe_list.append(epe.view(-1).numpy())  # Store as numpy for aggregation

    # Calculate mean EPE over the whole dataset
    epe = np.mean(np.concatenate(epe_list))
    print(f"Validation EPE: {epe:.6f}")

    return {'epe': epe}