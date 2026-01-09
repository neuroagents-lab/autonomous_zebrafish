import torch
import numpy as np
import tqdm
import sys


@torch.no_grad()
def validate_model(model, val_dataset, device):
    """
    Perform evaluation on a validation dataset passed as a parameter.
    """
    model.eval() 
    epe_list = []
    for i_batch, data_blob in enumerate(tqdm.tqdm(val_dataset, desc=f"Validation", leave=False, file=sys.stdout)):
        image1, image2, flow_gt = [x.to(device) for x in data_blob]
        flow_pr = model([image1, image2])

        # Compute End-Point Error (EPE)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=1).sqrt()  # EPE per pixel
        epe_list.append(epe.view(-1).cpu().numpy())  # Store as numpy for aggregation

    # Calculate mean EPE over the whole dataset
    epe = np.mean(np.concatenate(epe_list))
    
    return epe
