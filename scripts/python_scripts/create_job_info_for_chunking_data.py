import numpy as np
import os
import xarray
import pickle

from zfa.data_comparisons.utils import load_data_tensors
#from zfa.core.default_dirs import DATA_ROOT, BASE_DIR

# reece: having issues importing functions from other directories in project, so these are hardcoded for now
DATA_ROOT = "/data/group_data/neuroagents_lab/"
BASE_ROOT = "/data/user_data/rdkeller/"
BASE_DIR = os.path.join(BASE_ROOT, "zfa/")

def get_unit_ranges(data, processing_chunk_size):
    # assumed that last index is unit index

    num_units = data.shape[-1]  # get number of units
    num_chunks = int(num_units / processing_chunk_size)  # get processing chunk size

    num_residuals = (
        num_units - num_chunks * processing_chunk_size
    )  # get number of residual units

    # get start and stop unit indices
    units = []
    for i in range(num_chunks):
        units.append((i * processing_chunk_size, (i + 1) * processing_chunk_size))

    # include the residual units
    units.append(
        (
            num_chunks * processing_chunk_size,
            num_chunks * processing_chunk_size + num_residuals,
        )
    )

    return units


# build dictionary with info for processing chunks


def create_job_info_for_chunking(processing_chunk_size, transition):
    job_info_for_chunking = {}

    glial_trials, neural_trials = load_data_tensors(transition)
    ANIMALS = list(glial_trials.keys())
    for animal in ANIMALS:
        print(animal)
        job_info_for_chunking[animal] = {}
        job_info_for_chunking[animal]["glial_num_units"] = glial_trials[
            animal
        ].sizes["units"]
        job_info_for_chunking[animal]["neural_num_units"] = neural_trials[
            animal
        ].sizes["units"]

        # get units to process based on processing chunk size
        glial_units = get_unit_ranges(
            glial_trials[animal], processing_chunk_size=processing_chunk_size
        )
        glial_num_jobs = len(glial_units)

        neural_units = get_unit_ranges(
            neural_trials[animal],
            processing_chunk_size=processing_chunk_size,
        )
        neural_num_jobs = len(neural_units)

        job_info_for_chunking[animal]["glial_units"] = glial_units
        job_info_for_chunking[animal]["neural_units"] = neural_units

        job_info_for_chunking[animal]["glial_num_jobs"] = glial_num_jobs
        job_info_for_chunking[animal]["neural_num_jobs"] = neural_num_jobs

        job_info_for_chunking[animal][
            "glial_processing_chunk_size"
        ] = processing_chunk_size
        job_info_for_chunking[animal][
            "neural_processing_chunk_size"
        ] = processing_chunk_size

    return job_info_for_chunking


if __name__ == "__main__":
    import argparse

    print(" ---Parsing args---    ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--processing_chunk_size", type=int, default=100)
    parser.add_argument("--transition", type=str, default="a2p")
    args = parser.parse_args()

    # get job info for chunking
    job_info_for_chunking = create_job_info_for_chunking(args.processing_chunk_size, args.transition)

    # save job info
    print(" ---Saving dictionary--- ")
    with open(os.path.join(BASE_DIR, "job_info_for_chunking.pickle"), "wb") as handle:
        pickle.dump(job_info_for_chunking, handle, protocol=pickle.HIGHEST_PROTOCOL)
