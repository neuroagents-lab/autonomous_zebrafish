print(" ---Loading modules--- ")

# import torch

# assert torch.cuda.is_available(), print("Not on a compute node. Do not run anything.")

import numpy as np
import warnings


# import matplotlib.pyplot as plt
import os

# import xarray
import pickle

# reece: having issues importing functions from other directories in project, so these are hardcoded for now

#from zfa.core.default_dirs import (
#    BASE_DIR,
#    NEURAL_TRIALS_PATH,
#    GLIAL_TRIALS_PATH,
#    INTER_ANIMAL_RESULTS_DIR_2,
#)

DATA_ROOT = "/data/group_data/neuroagents_lab/"
BASE_ROOT = "/data/user_data/rdkeller/"
BASE_DIR = os.path.join(BASE_ROOT, "zfa/")
NEURAL_TRIALS_PATH = os.path.join(BASE_DIR, "neural_trials.pickle")
GLIAL_TRIALS_PATH = os.path.join(BASE_DIR, "glial_trials.pickle")
INTER_ANIMAL_RESULTS_DIR_2 = os.path.join(BASE_DIR, "inter_animal_results_81723/")


from utils import (
    load_data_tensors,
    get_units_from_job_info,
    get_save_name_from_args,
    build_param_lookup,
)

from sklearn.model_selection import KFold
from brainmodel_utils.core.constants import RIDGECV_ALPHA_CV
from brainmodel_utils.neural_mappers.utils import (
    generate_train_test_splits,
    convert_dict_to_tuple,
)
from brainmodel_utils.metrics.consistency import get_linregress_consistency
import itertools


def perform_cv(
    train_frac,
    num_splits,
    num_cv_splits,
    num_parallel_jobs,
    job_ID,
    source_animal,
    target_animal,
    source_cell_type,
    target_cell_type,
    num_bootstrap_iters=1000,
    metric="pearsonr",
):
    source_animal = ANIMALS[source_animal]
    target_animal = ANIMALS[target_animal]
    units = get_units_from_job_info(args)

    # get source and target data matrices
    if source_cell_type == "glial":
        source_resp = glial_brain_data[source_animal]
    else:
        source_resp = neural_brain_data[source_animal]

    if target_cell_type == "glial":
        target_resp = glial_brain_data[target_animal][
            :, :, units[job_ID][0] : units[job_ID][1]
        ]
    else:
        target_resp = neural_brain_data[target_animal][
            :, :, units[job_ID][0] : units[job_ID][1]
        ]

    print(" ---Running linear regression consistency--- ")

    results_list = []
    results = {}

    # can just add an outer loop here to go through target units and then dump the temp data

    if target_animal not in results.keys():
        results[target_animal] = {}

    print(
        f"    ---Analyzing target units {units[job_ID][0]} to {units[job_ID][1]}---   "
    )

    try:
        map_kwargs = {"map_type": "percentile"}
        # turn it into immutable tuple to store as a key
        map_kwargs_key = convert_dict_to_tuple(map_kwargs)
        results[target_animal][map_kwargs_key] = {}

        results[target_animal][map_kwargs_key][
            str(units[job_ID])
        ] = get_linregress_consistency(
            source=source_resp,
            target=target_resp,
            map_kwargs=map_kwargs,
            num_bootstrap_iters=num_bootstrap_iters,
            num_parallel_jobs=num_parallel_jobs,
            splits=None,
            train_frac=0.5,
            num_train_test_splits=5,
        )

    except ZeroDivisionError:
        return results

    print(results)

    results_list.append(results)

    return results


if __name__ == "__main__":
    import argparse

    print(" ---Parsing args---    ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--num-bootstrap-iters", type=int, default=1000)
    parser.add_argument("--num-cv-splits", type=int, default=5)
    parser.add_argument("--num-parallel-jobs", type=int, default=2)
    parser.add_argument("--source_cell_type", type=str, default="glial")
    parser.add_argument("--target_cell_type", type=str, default="glial")
    parser.add_argument("--source_animal", type=int, default=0)  # target animal ID
    parser.add_argument("--target_animal", type=int, default=0)  # target animal ID
    parser.add_argument("--job_ID", type=int, default=1)

    args = parser.parse_args()

    # load both neural and glial data
    neural_brain_data, glial_brain_data = load_data_tensors()

    # grab params and run cv
    params = build_param_lookup(args)
    ANIMALS = list(neural_brain_data.keys())
    curr_params = params["0"]
    results = perform_cv(**curr_params)

    # get save name
    save_name = get_save_name_from_args(args)

    # save the data
    with open(INTER_ANIMAL_RESULTS_DIR_2 + save_name, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
