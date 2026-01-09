import pickle
import os
import torch as th

# reece: having issues importing functions from other directories in project, so these are hardcoded for now

#from zfa.core.default_dirs import (
#    BASE_DIR,
#    NEURAL_TRIALS_PATH,
#    GLIAL_TRIALS_PATH,
#    INTER_ANIMAL_RESULTS_DIR,
#)

# BASE_ROOT = "/data/user_data/rdkeller/"
BASE_ROOT = "/data/group_data/neuroagents_lab/"
BASE_DIR = os.path.join(BASE_ROOT, "zfa/")

NEURAL_PASSIVE_TRIALS_PATH = os.path.join(BASE_DIR, "neural_passive_trials.pickle")
GLIAL_PASSIVE_TRIALS_PATH = os.path.join(BASE_DIR, "glial_passive_trials.pickle")
NEURAL_ACTIVE_TRIALS_PATH = os.path.join(BASE_DIR, "neural_active_trials.pickle")
GLIAL_ACTIVE_TRIALS_PATH = os.path.join(BASE_DIR, "glial_active_trials.pickle")

# WHITE_NOISE_TRIALS_PATH = os.path.join(BASE_DIR, "white_noise.pickle")
# GP0_TRIALS_PATH = os.path.join(BASE_DIR, "gp0.pickle")
# GP1_TRIALS_PATH = os.path.join(BASE_DIR, "gp1.pickle")
# BANG_BANG_TRIALS_PATH = os.path.join(BASE_DIR, "bang_bang.pickle")
# AVG0_TRIALS_PATH = os.path.join(BASE_DIR, "avg0.pickle")
# AVG1_TRIALS_PATH = os.path.join(BASE_DIR, "avg1.pickle")

BRAIN_MODEL_RESULTS_DIR = os.path.join(BASE_DIR, "brain_model_results/")

# VF_TRIALS_PATH = os.path.join(BRAIN_MODEL_RESULTS_DIR, "vf.pickle")
# VF_DIS_TRIALS_PATH = os.path.join(BASE_DIR, "vf_dis.pickle")
# VF_PROG_TRIALS_PATH = os.path.join(BASE_DIR, "vf_prog.pickle")
INTER_ANIMAL_RESULTS_DIR_2 = os.path.join(BASE_DIR, "inter_animal_results_81723/")

def load_model_tensors(transition, model_type):
    path = os.path.join(BRAIN_MODEL_RESULTS_DIR, model_type, f"{model_type}_{transition}.pickle")
    with open(path, "rb") as handle:
        model_data = pickle.load(handle)
    return model_data

def load_control_tensors(transition, model, source_animal, source_cell_type):
    if model == "PID":
        path = os.path.join(BASE_DIR, f"{model}_{source_cell_type}_{transition}.pickle")
    else: 
        path = os.path.join(BASE_DIR, f"{model}{source_animal}_{source_cell_type}_{transition}.pickle")
    with open(path, "rb") as handle:
        control_data = pickle.load(handle)
    return control_data

def load_data_tensors(transition):
    # loads both neural and glial tensors
    if transition == "passive":
        with open(GLIAL_PASSIVE_TRIALS_PATH, "rb") as handle:
            glial_trials = pickle.load(handle)
            glial_brain_data = glial_trials["tensors"]

        with open(NEURAL_PASSIVE_TRIALS_PATH, "rb") as handle:
            neural_trials = pickle.load(handle)
            neural_brain_data = neural_trials["tensors"]
            
    elif transition == "active":
        with open(GLIAL_ACTIVE_TRIALS_PATH, "rb") as handle:
            glial_trials = pickle.load(handle)
            glial_brain_data = glial_trials["tensors"]

        with open(NEURAL_ACTIVE_TRIALS_PATH, "rb") as handle:
            neural_trials = pickle.load(handle)
            neural_brain_data = neural_trials["tensors"]

    return glial_brain_data, neural_brain_data

def get_units_from_job_info(args):
    # load in job information containing units
    with open(os.path.join(BASE_DIR, "job_info_for_chunking.pickle"), "rb") as handle:
        job_info_for_chunking = pickle.load(handle)

    ANIMALS = list(job_info_for_chunking.keys())
    if args.target_cell_type == "neural":
        return job_info_for_chunking[ANIMALS[args.target_animal]]["neural_units"]
    else:
        return job_info_for_chunking[ANIMALS[args.target_animal]]["glial_units"]


def get_save_name_from_args(args):
    return f"source_cell_type={args.source_cell_type}_target_cell_type={args.target_cell_type}_jobID={args.job_ID}_source_animal={args.source_animal}_target_animal={args.target_animal}_inter-animal-consistency.pickle"


def build_param_lookup(args):
    param_lookup = {}

    key = 0

    param_lookup[str(key)] = {
        "train_frac": args.train_frac,
        "num_splits": args.num_splits,
        "num_cv_splits": args.num_cv_splits,
        "num_bootstrap_iters": args.num_bootstrap_iters,
        "num_parallel_jobs": args.num_parallel_jobs,
        "job_ID": args.job_ID,
        "source_animal": args.source_animal,
        "target_animal": args.target_animal,
        "source_cell_type": args.source_cell_type,
        "target_cell_type": args.target_cell_type,
    }
    key += 1

    return param_lookup

def downsample(x, n_out=20):
    T, F = x.shape
    base, rem = divmod(T, n_out) 
    block_sizes = [base + 1]*rem + [base]*(n_out - rem)
    idx = th.tensor([0] + list(th.cumsum(th.tensor(block_sizes), dim=0)))
    out = th.stack([x[idx[i]:idx[i+1]].mean(dim=0) for i in range(n_out)], dim=0)
    return out
    
