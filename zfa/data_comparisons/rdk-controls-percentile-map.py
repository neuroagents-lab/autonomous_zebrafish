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

from utils import (
    load_model_tensors,
    load_data_tensors,
    load_control_tensors,
    get_units_from_job_info,
    get_save_name_from_args,
    build_param_lookup,
    BRAIN_MODEL_RESULTS_DIR
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
    source_animal: str,
    target_animal: int,
    source_cell_type: str,
    target_cell_type: str,
    num_bootstrap_iters=10,
    metric="pearsonr",
    ):

    print("Source Animal: ", source_animal)
    print("Source Cell Type: ", source_cell_type)
    print("Target Animal: ", target_animal)
    print("Target Cell Type: ", target_cell_type)
    target_animal = ANIMALS[target_animal]
    units = get_units_from_job_info(args)

    if target_cell_type == "glial":
        target_resp = glial_brain_data[target_animal][
            :, :, units[job_ID][0] : units[job_ID][1]
        ]
    else:
        target_resp = neural_brain_data[target_animal][
            :, :, units[job_ID][0] : units[job_ID][1]
        ]

    print(" ---Running linear regression consistency--- ")

    results = {}
    # can just add an outer loop here to go through target units and then dump the temp data

    if target_animal not in results.keys():
        results[target_animal] = {}

    print(type(control_data), type(target_resp))
    print(control_data.shape, target_resp.shape)
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
            source=control_data,
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
    return results


if __name__ == "__main__":
    import argparse

    print(" ---Parsing args---    ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", type=str, default="active")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--num-bootstrap-iters", type=int, default=1000)
    parser.add_argument("--num-cv-splits", type=int, default=5)
    parser.add_argument("--num-parallel-jobs", type=int, default=2)
    parser.add_argument("--source_cell_type", type=str, default="glial")
    parser.add_argument("--target_cell_type", type=str, default="glial")
    parser.add_argument("--source_animal", type=int, default=0)  # source animal ID
    parser.add_argument("--target_animal", type=int, default=0)  # target animal ID
    parser.add_argument("--model", type=str, default="avg")  # model type
    parser.add_argument("--job_ID", type=int, default=1)
    
    args = parser.parse_args()

    assert args.source_cell_type == args.target_cell_type, "Source and target cell types must be the same for this analysis."

    model_type = args.model
    if model_type != "avg":
        assert args.source_animal == args.target_animal, "Source and target animals must be the same for this analysis."
        
    transition_type = args.transition
    glial_brain_data, neural_brain_data = load_data_tensors(transition_type)
    control_data = load_control_tensors(transition_type, model_type, args.source_animal, args.source_cell_type)
    ANIMALS = list(neural_brain_data.keys())

    # grab params and run cv
    params = build_param_lookup(args)
    curr_params = params["0"]
    results = perform_cv(**curr_params)

    # get save name
    save_name = get_save_name_from_args(args)

    # save the data
    save_path = os.path.join(BRAIN_MODEL_RESULTS_DIR, model_type, transition_type, args.target_cell_type, save_name)
    with open(save_path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
