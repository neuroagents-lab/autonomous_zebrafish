import os
import xarray
import pickle

from zfa.core.default_dirs import (
    BASE_DIR,
    NEURAL_TRIALS_PATH,
    GLIAL_TRIALS_PATH,
    INTER_ANIMAL_RESULTS_DIR,
)


def get_desired_value(job_info_for_chunking, args):
    ANIMALS = list(job_info_for_chunking.keys())

    return job_info_for_chunking[ANIMALS[args.animal]][args.cell_type]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # two options are sub-20170228-3_ses-20170228T165730_ophys and sub-20170228-4_ses-20170228T185002_ophys
    parser.add_argument("--animal", type=int, default=0)
    parser.add_argument("--cell_type", type=str, default="glial_num_jobs")

    # parser.add_argument("--job_number", type=int, default=1)

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, "job_info_for_chunking.pickle"), "rb") as handle:
        job_info_for_chunking = pickle.load(handle)

    print(get_desired_value(job_info_for_chunking, args))


# pre-compute the unit indices for the four datasets and number of parallel jobs, store in x-array
# read in number of jobs
#
