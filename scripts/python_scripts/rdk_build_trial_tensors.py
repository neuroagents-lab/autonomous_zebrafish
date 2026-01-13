from pynwb import NWBHDF5IO
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import xarray as xr
import pandas as pd

def normalize_series(series):
    # Ensure the series is in the correct shape for the MinMaxScaler (samples, features)
    reshaped_series = np.array(series).reshape(-1, 1)
    
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    normalized = scaler.fit_transform(reshaped_series)
    
    # Since our time series is a 1-D array, we can flatten the output to get it back into the original shape
    return normalized.flatten()

def get_neural_and_glial_trials(nwbfilepath, transition="passive", window_size=10):

    print('Reading file path...')
    io = NWBHDF5IO(nwbfilepath, mode="r", load_namespaces=True)
    nwbfile = io.read()
    
    print('Getting trial start and stop points...')
    glial_trace = nwbfile.processing['ophys']['DfOverF']['GliaDfOverF'].data
    neural_trace = nwbfile.processing['ophys']['DfOverF']['NeuronDfOverF'].data
    time_stamps = np.asarray(nwbfile.processing['ophys']['DfOverF']['GliaDfOverF'].timestamps)

    # activity_states = compute_activity_states(nwbfile)
    activity_states = nwbfile.processing['behavior']['ActivityStates'].to_dataframe()
    activity_states = activity_states[~(activity_states['state_type'] == 'transient')]
    behavior_state_types = activity_states['state_type'].to_numpy()
    behavior_start_times = activity_states['start_time'].to_numpy()
    behavior_stop_times = activity_states['stop_time'].to_numpy()

    # trial_types = nwbfile.trials['trial_type']
    # trial_start_times = np.asarray(nwbfile.trials['start_time'])
    # trial_stop_times = np.asarray(nwbfile.trials['stop_time'])

    print('Extracting glial and neural trials...')
    glial_trials = []
    neural_trials = []

    assert len(time_stamps) == glial_trace.shape[0], print('Wrong timestaps!')
    assert len(time_stamps) == neural_trace.shape[0], print('Wrong timestaps!')
    
    for i,behavior in enumerate(behavior_state_types):
        if behavior == transition:  # 'passive':
            # skip first trial as unknown transition
            if i == 0:
                continue
            # check if previous trial is contiguous and different state from current
            if np.abs(behavior_start_times[i] - behavior_stop_times[i-1]) > 0.1:
                continue
            if behavior_state_types[i-1] == behavior:
                continue

            start_ind = np.argmin(np.abs(time_stamps - behavior_start_times[i]))
            #end_ind = np.argmin(np.abs(glial_times- behavior_stop_times[i]))

            glial_trial  = glial_trace[start_ind-window_size:start_ind + window_size + 1,:]
            neural_trial = neural_trace[start_ind-window_size:start_ind + window_size + 1,:]

            #if glial_trial.shape == neural_trial.shape:#, print('Glial time series and neural time series not the same length!')
            if len(glial_trial) == window_size*2 + 1:
                glial_trials.append(glial_trial)
                neural_trials.append(neural_trial)
    print(f'Finished extraction... found {len(glial_trials)} trials.')
    
    return np.stack(glial_trials), np.stack(neural_trials)

def convert_np_to_xarray(nparray, window_size = 10):


    coords = {
        "trials": np.arange(nparray.shape[0]),  # replace with your actual coordinates
        "time": np.linspace(-window_size,window_size,nparray.shape[1]),    # replace with your actual coordinates
        "units": np.arange(nparray.shape[2])   # replace with your actual coordinates
    }

    # convert the numpy array to xarray
    xarray_from_nparray = xr.DataArray(nparray, dims=("trials", "time", "units"), coords = coords)

    return xarray_from_nparray

def save_neural_and_glial_pickle(glial_trials, neural_trials, filenames, save_dir, save_suffix="_trials"):
    glial_trials_xarray = [convert_np_to_xarray(gt) for gt in glial_trials]
    neural_trials_xarray = [convert_np_to_xarray(nt) for nt in neural_trials]

    save_keys = [fn.split('/')[-1].replace('.nwb','') for fn in filenames]
    neural_tensors = dict(zip(save_keys, neural_trials_xarray))
    glial_tensors = dict(zip(save_keys, glial_trials_xarray))

    neural_trial_dict = {'filenames': filenames, 'tensors': neural_tensors}
    glial_trial_dict = {'filenames': filenames, 'tensors': glial_tensors}

    #save the data
    with open(save_dir + f'glial{save_suffix}.pickle', 'wb') as handle:
        pickle.dump(glial_trial_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_dir + f'neural{save_suffix}.pickle', 'wb') as handle:
        pickle.dump(neural_trial_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Data saved to: ', save_dir + f'glial{save_suffix}.pickle', ', ', save_dir + f'neural{save_suffix}.pickle')

if __name__ == "__main__":
    sesh3 = '/data/group_data/neuroagents_lab/legacy_neural_datasets/zfa_data/000350/sub-20170228-3/sub-20170228-3_ses-20170228T165730_ophys.nwb'
    sesh4 = '/data/group_data/neuroagents_lab/legacy_neural_datasets/zfa_data/000350/sub-20170228-4/sub-20170228-4_ses-20170228T185002_ophys.nwb'

    filenames = [sesh3, sesh4]

    glial_passive_trials, neural_passive_trials = zip(*[get_neural_and_glial_trials(f, transition="passive", window_size=10) for f in filenames])
    save_neural_and_glial_pickle(
        glial_passive_trials, 
        neural_passive_trials, 
        filenames, 
        '/data/user_data/rdkeller/zfa/', 
        save_suffix="_passive_trials",
    )

    glial_active_trials, neural_active_trials = zip(*[get_neural_and_glial_trials(f, transition="active", window_size=10) for f in filenames])
    save_neural_and_glial_pickle(
        glial_active_trials, 
        neural_active_trials, 
        filenames, 
        '/data/user_data/rdkeller/zfa/', 
        save_suffix="_active_trials",
    )