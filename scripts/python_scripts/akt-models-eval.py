import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from zfa.models.optic_flow_architetures import Net_3_layers
from zfa.model_training.datasets import optic_flow_fetch_dataloader

class Args:
    def __init__(self):
        self.split = 'test'  
        self.save_checkpoint = False  
        self.num_epochs = 1  
        self.gpus = [0]  
        self.mixed_precision = False  
        self.clip = 1.0
        self.add_noise = False  
        self.batch_size = 1  
        self.num_workers = 8  
        self.size = 64  

args = Args()

def initialize_model(model_path):
    """ Load the model from the checkpoint """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net_3_layers(input_size=64)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device  

def fetch_sample_data(args, model, device, test_loader):
    """ Fetch new test samples in every run """
    try:
        sample_batch = next(iter(test_loader))  
    except StopIteration:
        print("End of test dataset, reloading...")
        test_loader = iter(optic_flow_fetch_dataloader(args)['test'])
        sample_batch = next(test_loader)

    image1, image2, flow_gt = [x.to(device) for x in sample_batch]

    with torch.no_grad():
        flow_pred = model([image1, image2])

    return image1, image2, flow_gt, flow_pred

def flow_to_color(flow):
    """ Convert optic flow to a color visualization. """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    hsv[..., 0] = ang * 180 / np.pi / 2  
    hsv[..., 1] = 255  
    hsv[..., 2] = np.clip(mag, 0, 255)  

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def visualize_comparison(models_dict, test_loader, num_iters=10, save_path="optic_flow_comparison.png"):
    """ 
    Compare multiple models across multiple iterations.
    
    - Rows: Different iterations (num_iters)
    - Columns: Each model and ground truth
    """
    num_models = len(models_dict)
    fig, axs = plt.subplots(num_iters, num_models + 1, figsize=(15, 3 * num_iters))

    # Set column titles only on the first row
    model_names = list(models_dict.keys()) + ["Ground Truth"]
    for col, title in enumerate(model_names):
        axs[0, col].set_title(title, fontsize=14)

    for i in range(num_iters):
        print(f"Iteration {i+1}/{num_iters}")

        # Fetch new data for this iteration
        image1, image2, flow_gt = None, None, None
        model_preds = {}

        for model_name, (model, device) in models_dict.items():
            image1, image2, flow_gt, flow_pred = fetch_sample_data(args, model, device, test_loader)
            model_preds[model_name] = flow_pred[0].detach().cpu().numpy().transpose(1, 2, 0)

        flow_gt = flow_gt[0].detach().cpu().numpy().transpose(1, 2, 0)  

        # Plot each model's prediction
        for col, (model_name, flow_pred) in enumerate(model_preds.items()):
            axs[i, col].imshow(flow_to_color(flow_pred))
            axs[i, col].axis("off")

        # Plot ground truth
        axs[i, num_models].imshow(flow_to_color(flow_gt))
        axs[i, num_models].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    best_models_dir = "/data/user_data/akirscht/zfa/models/"
    best_models_paths = [
        "optic_flow_64_lr0.1_wd0.0001_best.pth",
        "optic_flow_64_lr0.01_wd0.0001_best.pth",
        "optic_flow_64_lr0.01_wd0.001_best.pth"
    ]

    # Convert paths and load models
    models_dict = {}
    for model_path in best_models_paths:
        full_path = os.path.join(best_models_dir, model_path)
        model_name = os.path.basename(model_path).replace(".pth", "")
        models_dict[model_name] = initialize_model(full_path)

    # Load test dataloader
    dataloader = optic_flow_fetch_dataloader(args)
    if 'test' not in dataloader:
        raise KeyError("Test dataloader not found! Ensure test data is initialized correctly.")
    test_loader = iter(dataloader['test'])

    # Run 100 iterations, saving every 10
    total_iterations = 100
    iterations_per_file = 10
    num_files = total_iterations // iterations_per_file

    for file_idx in range(num_files):
        save_path = f"optic_flow_comparison_{file_idx+1}.png"
        print(f"Generating {save_path} with {iterations_per_file} iterations...")
        visualize_comparison(models_dict, test_loader, num_iters=iterations_per_file, save_path=save_path)
