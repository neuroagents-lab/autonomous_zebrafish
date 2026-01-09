import numpy as np
import torch
import time
import xarray as xr


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
