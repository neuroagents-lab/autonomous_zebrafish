import torch
import tqdm
from zfa.model_training.datasets import optic_flow_fetch_dataloader

def classify_direction(flow):
    """Classifies the flow into Up, Down, Left, or Right based on average direction."""
    avg_u = torch.mean(flow[0, :, :])  # Horizontal motion (u channel)
    avg_v = torch.mean(flow[1, :, :])  # Vertical motion (v channel)

    # Determine the dominant direction
    if abs(avg_u) > abs(avg_v):  # More movement in horizontal direction
        return "Right" if avg_u > 0 else "Left"
    else:  # More movement in vertical direction
        return "Down" if avg_v > 0 else "Up"

def process_flo_dataloader(args):
    """Processes all .flo files using dataloader and categorizes them."""
    categories = {"Up": 0, "Down": 0, "Left": 0, "Right": 0}

    # Load dataset using fetch_dataloader
    dataloaders = optic_flow_fetch_dataloader(args)
    train_loader = dataloaders['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_samples = 0  # Track total processed samples

    for batch in tqdm.tqdm(train_loader):
        _, _, flows = [x.to(device) for x in batch]  # Assuming dataset returns (image1, image2, flow)

        try:
            for flow in flows:  # Process each flow sample separately
                direction = classify_direction(flow)
                categories[direction] += 1
                total_samples += 1  # Increment count per sample
        except Exception as e:
            print(f"Skipping batch due to error: {e}")

    # Print results
    print(f"\nTotal samples processed: {total_samples} / {583710}")
    print("\nClassification Results:")
    for direction, count in categories.items():
        print(f"{direction}: {count}")

    return categories

# Example usage
class Args:
    def __init__(self):
        self.split = 'train'  # Choose the appropriate split
        self.batch_size = 128   # Adjust as needed
        self.num_workers = 32
        self.size = 64  # Ensure the size matches training

args = Args()
process_flo_dataloader(args)
