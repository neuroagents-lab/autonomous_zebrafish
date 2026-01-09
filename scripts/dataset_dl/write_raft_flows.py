import os
import torch
import argparse
from tqdm import tqdm

from zfa.models.raft.raft import RAFT
from zfa.models.raft.utils.utils import InputPadder, forward_interpolate
from zfa.models.paths import PATHS
from zfa.model_training.datasets import FramePairDataset
from zfa.core.default_dirs import MVK_FRAME_DIR, MVK_FLOW_DIR
from zfa.core.frame_utils import writeFlow


def get_sequence_name(image_paths):
    # Check if the input is a tuple of length 2
    if not isinstance(image_paths, tuple) or len(image_paths) != 2:
        raise ValueError("Input must be a tuple of length 2")

    # Check if the sequence names are the same for both image paths
    directory1 = os.path.dirname(image_paths[0])
    directory2 = os.path.dirname(image_paths[1])
    sequence_name1 = os.path.basename(directory1)
    sequence_name2 = os.path.basename(directory2)
    if sequence_name1 != sequence_name2:
        raise ValueError("Sequence names in the input tuple are not the same")

    # Extract the sequence name
    return sequence_name1


def get_frame_name(image_path):
    # Extract the filename stem from the first image path
    filename = os.path.basename(image_path)
    frame_name = os.path.splitext(filename)[0]
    return frame_name


def get_flow_path(image_paths):
    # Check if the input is a tuple of length 2
    if not isinstance(image_paths, tuple) or len(image_paths) != 2:
        raise ValueError("Input must be a tuple of length 2")

    frame_name = get_frame_name(image_paths[0])
    flow_name = frame_name.replace("frame", "flow")
    flow_name += ".flo"
    # Replace directory part with MVK_FLOW_DIR
    flow_dir = os.path.dirname(image_paths[0]).replace(MVK_FRAME_DIR, MVK_FLOW_DIR)

    # Construct the full path of the flow file
    flow_path = os.path.join(flow_dir, flow_name)

    return flow_dir, flow_path


# iterations matching sintel eval: https://github.com/princeton-vl/RAFT/blob/master/evaluate.py#L96,
# as well as warm starting with flow of previous frame of same video: https://github.com/princeton-vl/RAFT/blob/master/evaluate.py#L184
@torch.no_grad()
def evaluate_mvk(model, mode, iters=32, warm_start=True):
    assert mode in ["train", "val", "test"]

    model.eval()
    # returns consecutive pairs of images, which we pass to RAFT
    dataset = FramePairDataset(root_dir=os.path.join(MVK_FRAME_DIR, mode))
    assert len(dataset) > 0

    # logic adapted from: https://github.com/princeton-vl/RAFT/blob/master/evaluate.py#L28-L49
    flow_prev, sequence_prev = None, None
    for idx in tqdm(range(len(dataset))):
        image1, image2 = dataset[idx]
        curr_image_paths = dataset.image_list[idx]
        sequence = get_sequence_name(curr_image_paths)
        # reinitialize flow for each video sequence
        if sequence != sequence_prev:
            flow_prev = None

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(
            image1, image2, iters=iters, flow_init=flow_prev, test_mode=True
        )
        # since batch size is 1, we index into the batch dimension
        # originally, it is 2 x h x w, and then it becomes h x w x 2
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        if warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        # write out the computed flow
        flow_dir, flow_path = get_flow_path(curr_image_paths)
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)
        writeFlow(flow_path, flow)

        sequence_prev = sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAFT on MVK frames")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="Number of iterations for flow estimation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset mode",
    )
    parser.add_argument(
        "--model", default=PATHS["raft_sintel"], help="restore checkpoint"
    )
    parser.add_argument("--dataset", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # logic adapted from: https://github.com/princeton-vl/RAFT/blob/master/evaluate.py#L178-L187
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    with torch.no_grad():
        evaluate_mvk(model.module, mode=args.mode)
