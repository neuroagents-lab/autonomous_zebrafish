import os


# from rl_zoo3.utils import is_mac
def is_mac():
    return False


#DATA_ROOT = "/om2/group/fiete/zfa/"
DATA_ROOT = "/data/group_data/neuroagents_lab/"

DATA_BASE_DIR = os.path.join(DATA_ROOT, "000350/")

#BASE_ROOT = "/om2/group/yanglab/"
BASE_ROOT = "/data/user_data/rdkeller/"

if is_mac():
    BASE_ROOT = os.path.expanduser("~/")
BASE_DIR = os.path.join(BASE_ROOT, "zfa/")

BASE_ROOT_2 = (
    "/om/weka/yanglab/anayebi/"  # only user directories are allowed modification in /om
)
BASE_DIR_2 = os.path.join(BASE_ROOT_2, "zfa/")

BASE_ROOT_3 = "/om/weka/yanglab/leokoz8/"
BASE_DIR_3 = os.path.join(BASE_ROOT_3, "zfa/")

# AKT
BASE_ROOT_4 = "/data/user_data/akirscht/"
BASE_DIR_4 = os.path.join(BASE_ROOT_4, "zfa/")


#INTER_ANIMAL_RESULTS_DIR = os.path.join(BASE_DIR_3, "inter_animal_results/")
INTER_ANIMAL_RESULTS_DIR = os.path.join(BASE_DIR, "inter_animal_results/")

#INTER_ANIMAL_RESULTS_DIR_2 = os.path.join(BASE_DIR_3, "inter_animal_results_81723/")
INTER_ANIMAL_RESULTS_DIR_2 = os.path.join(BASE_DIR, "inter_animal_results_81723/")


VOLUSEG_BASE_DIR = os.path.join(BASE_DIR, "voluseg/")

ANTS_DIR = os.path.join(BASE_ROOT, "ants/install/bin/")

UNITY_BUILDS_DIR = os.path.join(BASE_ROOT, "unity_builds/zfa/")

MODEL_CKPT_DIR = os.path.join(BASE_DIR_2, "model_ckpts/")

NEURAL_TRIALS_PATH = os.path.join(BASE_DIR, "neural_trials.pickle")

GLIAL_TRIALS_PATH = os.path.join(BASE_DIR, "glial_trials.pickle")

PRETRAINING_DATASET_DIR = os.path.join(DATA_ROOT, "training_datasets/zfa_pretraining_data/")
MVK_RAW_DIR = os.path.join(
    PRETRAINING_DATASET_DIR, "raw/hkust-vgd.ust.hk/MarineVideoKit/videos/"
)

MVK_FRAME_DIR = os.path.join(PRETRAINING_DATASET_DIR, "processed/MarineVideoKit/")
MVK_FLOW_DIR = os.path.join(MVK_FRAME_DIR, "processed/MarineVideoKit_flows/")

SCRATCH_FRAME_DIR = "/scratch/akirscht/marine_video_kit_50"
SCRATCH_FLOW_DIR = "/scratch/akirscht/marine_video_kit_50/processed/MarineVideoKit_flows" 


OPTIC_FLOW_CHKP_DIR = os.path.join(BASE_DIR_4, "checkpoints/")
OPTIC_FLOW_LOGS_DIR = "/home/akirscht/tensorboard_logs/"
OPTIC_FLOW_MODEL_DIR = os.path.join(BASE_DIR_4, "models/")

