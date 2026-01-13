import numpy as np
import torch
import time
import xarray as xr

from brainmodel_utils.core.constants import RIDGECV_ALPHA_CV
from brainmodel_utils.neural_mappers import PercentileNeuralMap
from brainmodel_utils.neural_mappers.utils import (
    generate_train_test_splits,
    convert_dict_to_tuple,
)

from brainmodel_utils.metrics.consistency import get_linregress_consistency


import time


# Ensure PyTorch is using a GPU
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot find a GPU!")

device = torch.device("cuda:0")


map_kwargs = {"map_type": "percentile"}
# turn it into immutable tuple to store as a key
map_kwargs_key = convert_dict_to_tuple(map_kwargs)

dim_trials = 5
dim_times = 10
dim_units = 100
window_size = 20


source_resp = np.random.normal(0, 1, size=(dim_trials, dim_times, dim_units))
target_resp = np.random.normal(0, 1, size=(dim_trials, dim_times, dim_units))


coords = {
    "trials": np.arange(dim_trials),  # replace with your actual coordinates
    "time": np.linspace(
        -window_size, window_size, dim_times
    ),  # replace with your actual coordinates
    "units": np.arange(dim_units),  # replace with your actual coordinates
}

source_resp = xr.DataArray(source_resp, dims=("trials", "time", "units"), coords=coords)
target_resp = xr.DataArray(target_resp, dims=("trials", "time", "units"), coords=coords)


# Store the start time
start_time = time.time()

get_linregress_consistency(
    source=source_resp,
    target=target_resp,
    map_kwargs=map_kwargs,
    num_bootstrap_iters=1,
    num_parallel_jobs=1,
    splits=None,
    train_frac=0.5,
    num_train_test_splits=5,
)

# Time the execution of the function
# Store the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
