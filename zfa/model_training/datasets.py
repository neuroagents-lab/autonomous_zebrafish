import os
import numpy as np
import random
import torch
import torch.utils.data as data
from glob import glob
import os.path as osp
from PIL import Image
import sys
import tqdm
import time
import cv2
from torch.utils.data import DataLoader

from zfa.core import frame_utils, default_dirs as dirs
from zfa.model_training.augmentor import FlowAugmentor, SparseFlowAugmentor, FlowAugmentorOpticFlow
#from ptutils.core.utils import set_seed

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


"""
class FramePairDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_list = []
        self.transform = transform
        self.root_dir = root_dir
        self.init_seed = False

        # Load all image sequences
        for scene in sorted(os.listdir(root_dir)):
            scene_path = osp.join(root_dir, scene)
            images = sorted(glob(osp.join(scene_path, "*.png")))
            for i in range(len(images) - 1):
                self.image_list.append((images[i], images[i + 1]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # sets the seed automatically once getitem is called
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                set_seed(worker_info.id)
                self.init_seed = True

        img1_path, img2_path = self.image_list[index]
        img1 = frame_utils.read_gen(img1_path)
        img2 = frame_utils.read_gen(img2_path)

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        img1 = np.array(img1).astype(np.uint8)
        img1 = img1[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()

        img2 = np.array(img2).astype(np.uint8)
        img2 = img2[..., :3]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return img1, img2
"""

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(
        self, aug_params=None, split="training", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))


class FlyingChairs(FlowDataset):
    def __init__(
        self, aug_params=None, split="train", root="datasets/FlyingChairs_release/data"
    ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, "*.ppm")))
        flows = sorted(glob(osp.join(root, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt("chairs_split.txt", dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(
        self, aug_params=None, root="datasets/FlyingThings3D", dstype="frames_cleanpass"
    ):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ["left"]:
            for direction in ["into_future", "into_past"]:
                image_dirs = sorted(glob(osp.join(root, dstype, "TRAIN/*/*")))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, "optical_flow/TRAIN/*/*")))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, "*.png")))
                    flows = sorted(glob(osp.join(fdir, "*.pfm")))
                    for i in range(len(flows) - 1):
                        if direction == "into_future":
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == "into_past":
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split="training", root="datasets/KITTI"):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == "testing":
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1k"):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(osp.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(osp.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1

class OpticFlowDataset(data.Dataset):
    def __init__(self, root_dir= dirs.MVK_FRAME_DIR, flow_dir= dirs.MVK_FLOW_DIR, split='train', aug_params=None, size=16):  
        """ Initialize the dataset """

        self.frame_dir = os.path.join(root_dir, split)
        self.flow_dir = os.path.join(flow_dir, split)
        self.image_list = []
        self.flow_list = []
        self.init_seed = False
        self.augmentor = None
        self.size = size

        # Load all image sequences
        for scene in tqdm.tqdm(sorted(os.listdir(self.frame_dir)), desc="Loading dataset", file=sys.stdout):
            scene_path = osp.join(self.frame_dir, scene)
            images = sorted(glob(osp.join(scene_path, "*.png")))
            flow_path = osp.join(self.flow_dir, scene)
            flows = sorted(glob(osp.join(flow_path, "*.flo")))
            
            for i in range(len(images) - 1):
                self.image_list.append((images[i], images[i + 1]))
                self.flow_list.append(flows[i])
            
        if aug_params:
            self.augmentor = FlowAugmentorOpticFlow(**aug_params)

    def __len__(self):
        """ Return the number of frame pairs and flow files """
        return len(self.flow_list)

    def __getitem__(self, idx):
        """ Get a frame pair and corresponding flow file """
        # sets the seed automatically once getitem is called
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                set_seed(worker_info.id)
                self.init_seed = True

        img1_path, img2_path = self.image_list[idx]
        img1 = frame_utils.read_gen(img1_path)
        img2 = frame_utils.read_gen(img2_path)
        flow_path = self.flow_list[idx]
        flow = frame_utils.read_gen(flow_path)
        #flow = np.random.rand(*(480, 853, 2))  # Random flow
        if self.augmentor:
            img1, img2, flow = self.augmentor(img1, img2, flow)
        else:
            # Convert to numpy arrays
            img1 = np.ascontiguousarray(img1)
            img2 = np.ascontiguousarray(img2)
            flow = np.ascontiguousarray(flow)
        
        # Resize frames and flow
        scale_x = self.size / img1.shape[1]
        scale_y = self.size / img1.shape[0]

        img1 = cv2.resize(img1, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        flow[:, :, 0] *= scale_x # Scale vector magnitudes too
        flow[:, :, 1] *= scale_y # Scale vector magnitudes too
        
        # Convert to torch tensors
        img1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float()  # [C, H, W]
        img2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float()  # [C, H, W]
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()  # [2, H, W]
        

        return img1, img2, flow

def optic_flow_fetch_dataloader(args):
    """ Function to return DataLoaders for train, val, and test datasets """
    # Define augmentation parameters for training
    aug_params_train = aug_params = {
        
        'crop_size': (256, 448), 
        'min_scale': -0.2,        # Scale down to 80% of the original size
        'max_scale': 0.2,         # Scale up to 120% of the original size
        'do_flip': True      
    }
    # Initialize datasets based on the split
    print("Fetching train dataset")
    train_dataset = OpticFlowDataset(split='train',aug_params=aug_params_train, size=args.size)
    print("Fetching val dataset")
    val_dataset = OpticFlowDataset(split='val')


    # Initialize DataLoaders for each split
    print("Fetching train dataloader")
    train_dataloader =  DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=args.num_workers,
            pin_memory=True
        )
    print("Fetching val dataloader")
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Do not shuffle for validation
            num_workers=args.num_workers,
            pin_memory=True
        )
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}
    return data_loaders


def fetch_dataloader(args, TRAIN_DS="C+T+K+S+H"):
    """Create the data loader for the corresponding trainign set"""

    if args.stage == "chairs":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.1,
            "max_scale": 1.0,
            "do_flip": True,
        }
        train_dataset = FlyingChairs(aug_params, split="training")

    elif args.stage == "things":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.4,
            "max_scale": 0.8,
            "do_flip": True,
        }
        clean_dataset = FlyingThings3D(aug_params, dstype="frames_cleanpass")
        final_dataset = FlyingThings3D(aug_params, dstype="frames_finalpass")
        train_dataset = clean_dataset + final_dataset

    elif args.stage == "sintel":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.6,
            "do_flip": True,
        }
        things = FlyingThings3D(aug_params, dstype="frames_cleanpass")
        sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")
        sintel_final = MpiSintel(aug_params, split="training", dstype="final")

        if TRAIN_DS == "C+T+K+S+H":
            kitti = KITTI(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.3,
                    "max_scale": 0.5,
                    "do_flip": True,
                }
            )
            hd1k = HD1K(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.5,
                    "max_scale": 0.2,
                    "do_flip": True,
                }
            )
            train_dataset = (
                100 * sintel_clean
                + 100 * sintel_final
                + 200 * kitti
                + 5 * hd1k
                + things
            )

        elif TRAIN_DS == "C+T+K/S":
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.stage == "kitti":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.4,
            "do_flip": False,
        }
        train_dataset = KITTI(aug_params, split="training")

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    return train_loader


class SyntheticOpticFlowDataset(data.Dataset):
    def __init__(self, num_samples=1184, image_size=(3, 64, 64), flow_size=(2, 64, 64), aug_params=None):
        """
        A synthetic dataset to mimic optic flow data.
        :param num_samples: Number of samples in the dataset.
        :param image_size: Size of the synthetic images (C, H, W).
        :param flow_size: Size of the synthetic flow (C, H, W).
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.flow_size = flow_size

        if aug_params:
            self.augmentor = FlowAugmentorOpticFlow(**aug_params)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic data
        img1 = torch.rand(*self.image_size) * 255.0  # Random image 1
        img2 = torch.rand(*self.image_size) * 255.0  # Random image 2
        flow = torch.rand(*self.flow_size)  # Random flow
        
        return img1, img2, flow
    
def optic_flow_fetch_dataloader_synthetic(args):
     # Define augmentation parameters for training
    aug_params_train = aug_params = {
        
        'crop_size': (256, 448), 
        'min_scale': -0.2,        # Scale down to 80% of the original size
        'max_scale': 0.2,         # Scale up to 120% of the original size
        'do_flip': True,         
    }

    train_dataset = SyntheticOpticFlowDataset(num_samples=1147551, 
                                              image_size=(3, 200, 100), 
                                              flow_size=(2, 200, 100),
                                              aug_params=aug_params_train)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )

    val_dataset = SyntheticOpticFlowDataset(num_samples=5632, 
                                            image_size=(3, 200, 100), 
                                            flow_size=(2, 200, 100))

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    return {'train': train_dataloader, 'val': val_dataloader}
    