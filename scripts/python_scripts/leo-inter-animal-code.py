print("Loading modules...")

import torch

#assert torch.cuda.is_available(), print("Not on a compute node. Do not run anything.")

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray
import pickle


# from zfa.core.default_dirs import DATA_ROOT

# from dandi.dandiapi import DandiAPIClient
# from pynwb import NWBHDF5IO
# from nwbwidgets import nwb2widget


from sklearn.model_selection import KFold
from brainmodel_utils.core.constants import RIDGECV_ALPHA_CV
from brainmodel_utils.neural_mappers.utils import (
    generate_train_test_splits,
    convert_dict_to_tuple,
)
from brainmodel_utils.metrics.consistency import get_linregress_consistency
import itertools

# reece: changed dirs
#glial_save_dir = "/om2/group/yanglab/zfa/"
#neural_save_dir = "/om2/group/yanglab/zfa/"

BASE_ROOT = "/data/user_data/rdkeller/"
BASE_DIR = os.path.join(BASE_ROOT, "zfa/")

glial_save_dir = BASE_DIR
neural_save_dir = BASE_DIR

print("Loading data tensors...")

# check that I can load the stored data back
with open(glial_save_dir + "glial_trials.pickle", "rb") as handle:
    glial_trials = pickle.load(handle)

with open(neural_save_dir + "neural_trials.pickle", "rb") as handle:
    neural_trials = pickle.load(handle)

brain_data = glial_trials["tensors"]
ANIMALS = brain_data.keys()

print("Genrating splits...")

splits = generate_train_test_splits(num_stim=20, num_splits=5, train_frac=0.5)
num_cv_splits = 5

results_list = []
for i, s in enumerate(splits):
    results = {}
    print(f"Doing split {i}")
    for animal_pair in itertools.permutations(ANIMALS, r=2):
        source_animal = animal_pair[0]
        target_animal = animal_pair[1]
        target_resp = brain_data[target_animal]
        source_resp = brain_data[source_animal]

        target_resp_train = target_resp.isel(time=s["train"])
        source_resp_train = source_resp.isel(time=s["train"])

        kf = KFold(n_splits=num_cv_splits)

        cv_splits = []
        for cv_train_idx, cv_val_idx in kf.split(
            X=source_resp_train.mean(dim="trials", skipna=True)
        ):
            cv_splits.append({"train": cv_train_idx, "test": cv_val_idx})

        if target_animal not in results.keys():
            results[target_animal] = {}

        for alpha in RIDGECV_ALPHA_CV:
            map_kwargs = {
                "map_type": "sklinear",
                "map_kwargs": {
                    "regression_type": "Ridge",
                    "regression_kwargs": {"alpha": alpha},
                },
            }
            # turn it into immutable tuple to store as a key
            map_kwargs_key = convert_dict_to_tuple(map_kwargs)
            results[target_animal][map_kwargs_key] = get_linregress_consistency(
                source=source_resp_train,
                target=target_resp_train,
                map_kwargs=map_kwargs,
                num_bootstrap_iters=10,
                num_parallel_jobs=5,
                splits=cv_splits,
            )

        results_list.append(results)
    print(results)
