import os
import pprint
import shutil
import voluseg
import h5py
from pynwb import NWBHDF5IO

from zfa.core.default_dirs import DATA_BASE_DIR, ANTS_DIR, VOLUSEG_BASE_DIR


def run_voluseq(dir_input, dir_output, num_tp):
    assert num_tp > 0
    # set and save parameters
    parameters0 = voluseg.parameter_dictionary()
    parameters0["dir_ants"] = ANTS_DIR
    parameters0["dir_input"] = dir_input
    parameters0["dir_output"] = dir_output
    # TODO: if we want to use the same parameters for all files, we should set them here to change
    parameters0["registration"] = "high"
    parameters0["diam_cell"] = 5.0
    parameters0["f_volume"] = 2.0
    parameters0["timepoints"] = num_tp
    voluseg.step0_process_parameters(parameters0)

    # load and print parameters
    filename_parameters = os.path.join(parameters0["dir_output"], "parameters.pickle")
    parameters = voluseg.load_parameters(filename_parameters)
    pprint.pprint(parameters)

    print("process volumes.")
    voluseg.step1_process_volumes(parameters)

    print("align volumes.")
    voluseg.step2_align_volumes(parameters)

    print("mask volumes.")
    voluseg.step3_mask_volumes(parameters)

    print("detect cells.")
    voluseg.step4_detect_cells(parameters)

    print("clean cells.")
    voluseg.step5_clean_cells(parameters)


def write_to_hdf5(nwb_data, h5_base_dir):
    assert nwb_data.ndim == 4  # time, y, x, z
    num_tp = nwb_data.shape[0]
    num_digits = len(str(num_tp))
    for t in range(num_tp):
        assert nwb_data[t].ndim == 3
        # z, y, x, the format that voluseg expects
        curr_nwb = nwb_data[t].transpose(2, 0, 1)
        tp_name = "{:0>{}}".format(str(t), num_digits + 1)
        h5_filename = f"TM{tp_name}.h5"
        h5_filepath = os.path.join(h5_base_dir, h5_filename)
        with h5py.File(h5_filepath, "w") as f:
            f.create_dataset("default", data=curr_nwb)
    return num_tp


def get_nwb_series_keys(nwbfile):
    nwb_keys = [k for k in nwbfile.acquisition.keys() if k.endswith("TwoPhotonSeries")]
    assert len(nwb_keys) > 0
    return nwb_keys


def process_file(nwb_filename):
    assert nwb_filename.endswith(".nwb")
    nwb_filepath = os.path.join(DATA_BASE_DIR, nwb_filename)
    assert os.path.isfile(nwb_filepath)
    with NWBHDF5IO(nwb_filepath, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
    nwb_series_keys = get_nwb_series_keys(nwbfile)
    for k in nwb_series_keys:  # could be Glia & Neurons, or just Glia, or just Neurons
        print(f"Running voluseg for file {nwbfile} and key {k}")
        curr_base_dir = os.path.join(
            VOLUSEG_BASE_DIR, nwb_filename.replace(".nwb", "").replace("/", "+"), k
        )
        curr_input_dir = os.path.join(curr_base_dir, "input/")
        os.mkdir(curr_input_dir)
        curr_output_dir = os.path.join(curr_base_dir, "output/")
        os.mkdir(curr_output_dir)
        num_tp = write_to_hdf5(
            nwb_data=nwbfile.acquisition[k].data, h5_base_dir=curr_input_dir
        )
        run_voluseq(dir_input=curr_input_dir, dir_output=curr_output_dir, num_tp=num_tp)
        # delete hdf5 files
        shutil.rmtree(curr_input_dir)
        print(f"Finished voluseg for file {nwbfile} and key {k}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb_filenames", type=str, default=None, required=True)
    args = parser.parse_args()

    for nwb_filename in args.nwb_filenames.split(","):
        process_file(nwb_filename)
