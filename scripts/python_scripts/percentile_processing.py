from tqdm import tqdm
import numpy as np
from itertools import groupby
import re
import os
import pickle

def parse_string(s):
    # Use regex to extract all key=value patterns
    #matches = re.findall(r'([a-zA-Z0-9_]+)=([a-zA-Z0-9]+)', s)
    matches = re.findall(r'([a-zA-Z0-9_]+)=([^_]+)', s)

    # Convert matches into a dictionary
    result = dict(matches)

    # Convert values that are digits into integers
    for key, value in result.items():
        if value.isdigit():
            result[key] = int(value)

    # Adjust keys to remove undesired underscores and adjust 'jobID' to 'job_ID'
    keys_to_adjust = ['_target_cell_type', '_jobID', '_source_animal', '_target_animal']
    adjustments = {
        '_target_cell_type': 'target_cell_type',
        '_jobID': 'job_ID',
        '_source_animal': 'source_animal',
        '_target_animal': 'target_animal'
    }
    for old_key, new_key in adjustments.items():
        if old_key in result:
            result[new_key] = result.pop(old_key)

    return result

def contains_substring(s,sub_string = "source_cell_type"):
    return sub_string in s

class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def pickle_load(file_path):
    with open(file_path, 'rb') as handle:
        data = NumpyUnpickler(handle).load()
    return data

# def pickle_load(file_path):
#     with open(file_path, 'rb') as handle:
#         data = pickle.load(handle)
#     return data

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, value)
            yield from recursive_items(value)
        else:
            yield (key, value)

def get_keys_for_data(data):
    keys_for_data = []
    for key, value in recursive_items(data):
        keys_for_data.append(key)
    return keys_for_data

def group_dictionaries(dicts):
    # First, sort the dictionaries based on the key fields
    key_fields = ["source_cell_type", "target_cell_type", "source_animal", "target_animal"]
    sorted_dicts = sorted(dicts, key=lambda d: tuple(d[k] for k in key_fields))

    # Then, group the dictionaries by the key fields
    grouped_dicts = {}
    for key, group in groupby(sorted_dicts, key=lambda d: tuple(d[k] for k in key_fields)):
        grouped_dicts[key] = list(group)

    return grouped_dicts

def extract_file_paths(directory):
    file_dicts = []

    # Loop through each file in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the full path of the file
            #full_path = os.path.join(root, file)
            if contains_substring(file,sub_string = "source_cell_type"):
                file_dict = parse_string(file)
                file_dict['filename'] = file
                file_dicts.append(file_dict)
    return file_dicts

def get_consistencies(INTER_ANIMAL_RESULTS_DIR):
    file_dicts = extract_file_paths(INTER_ANIMAL_RESULTS_DIR)
    grouped_dicts = group_dictionaries(file_dicts)
    condition_and_consistencies = {}
    key_list = list(grouped_dicts.keys())

    # Loop through each inter-animal/animal-control comparison
    for key in key_list:        
        consistency = []

        # Loop through each job for that comparison
        for job_dict in tqdm(grouped_dicts[key]):
            full_file_path = INTER_ANIMAL_RESULTS_DIR + job_dict['filename']
            data = pickle_load(full_file_path)

            kfd = get_keys_for_data(data)
            chunk_consistency = data[kfd[0]][kfd[1]][kfd[2]]['test']['r_xy_n_sb']
            consistency.append(chunk_consistency)
        condition_and_consistencies[key]=np.concatenate(consistency, axis=-1)
    return condition_and_consistencies