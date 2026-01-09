import os
import itertools
import numpy as np

from sklearn.model_selection import KFold
from brainmodel_utils.core.constants import RIDGECV_ALPHA_CV
from brainmodel_utils.neural_mappers.utils import (
    generate_train_test_splits,
    convert_dict_to_tuple,
)
from brainmodel_utils.metrics.consistency import get_linregress_consistency

# from rel_inference.core.default_dirs import (
#     NEURAL_FIT_CV_SEARCH_DIR,
#     NEURAL_RESP_PACKAGED,
# )
# from rel_inference.core.constants import BRAIN_AREAS, ANIMALS
# from rel_inference.core.utils import get_filename, get_packaged_data_filename
# from rel_inference.neural_data.utils import get_single_area_common_stim_resp


def build_param_lookup(args):
    if args.brain_areas is None:
        brain_areas = BRAIN_AREAS
    else:
        brain_areas = args.brain_areas.split(",")

    param_lookup = {}
    key = 0
    for brain_area in brain_areas:
        param_lookup[str(key)] = {
            "brain_area": brain_area,
            "train_frac": args.train_frac,
            "time_mode": args.time_mode,
            "temporal": args.temporal,
            "trial_threshold": args.trial_threshold,
            "downsample_rate": args.downsample_rate,
            "start_offset_sec": args.start_offset_sec,
            "trial_frac_lower_bound": args.trial_frac_lower_bound,
            "additional_target_offset_sec": args.additional_target_offset_sec,
            "num_splits": args.num_splits,
            "num_cv_splits": args.num_cv_splits,
            "num_bootstrap_iters": args.num_bootstrap_iters,
            "num_parallel_jobs": args.num_parallel_jobs,
            "enforce_finite_mean": True if not args.no_finite_mean_filt else False,
        }
        key += 1

    return param_lookup


ANIMALS = ["fish1", "fish2"]


def perform_cv(
    brain_area,
    train_frac,
    num_splits,
    num_cv_splits,
    num_parallel_jobs,
    temporal=False,
    trial_threshold=360,
    downsample_rate=None,
    start_offset_sec=0.0,
    trial_frac_lower_bound=0.5,
    additional_target_offset_sec=0.2,
    num_bootstrap_iters=1000,
    metric="pearsonr",
    enforce_finite_mean=True,
    time_mode="target_gocue",
):
    assert len(ANIMALS) == 2  # makes code below simpler

    # packaged_fn = get_packaged_data_filename(
    #     brain_area=brain_area,
    #     time_mode=time_mode,
    #     temporal=temporal,
    #     collapse_temporal=True,
    #     trial_threshold=trial_threshold,
    #     start_offset_sec=start_offset_sec,
    #     downsample_rate=downsample_rate,
    #     enforce_finite_mean=enforce_finite_mean,
    #     trial_frac_lower_bound=trial_frac_lower_bound,
    #     additional_target_offset_sec=additional_target_offset_sec,
    # )
    # packaged_fn = os.path.join(NEURAL_RESP_PACKAGED, packaged_fn)
    if os.path.exists(packaged_fn):
        print(f"Loading packaged data from {packaged_fn}")
        brain_data = np.load(packaged_fn, allow_pickle=True)["arr_0"][()]
    # else:
    #     if temporal:
    #         brain_data = get_single_area_common_stim_resp(
    #             brain_area=brain_area,
    #             time_mode=time_mode,
    #             temporal=temporal,
    #             collapse_temporal=True,
    #             trial_threshold=trial_threshold,
    #             start_offset_sec=start_offset_sec,
    #             downsample_rate=downsample_rate,
    #             enforce_finite_mean=enforce_finite_mean,
    #         )
    #     else:
    #         brain_data = get_single_area_common_stim_resp(
    #             brain_area=brain_area,
    #             time_mode=time_mode,
    #             temporal=temporal,
    #             enforce_finite_mean=enforce_finite_mean,
    #             trial_frac_lower_bound=trial_frac_lower_bound,
    #             additional_target_offset_sec=additional_target_offset_sec,
    #         )
    if temporal:
        num_stim = len(brain_data[ANIMALS[0]].stim_time)
    else:
        num_stim = len(brain_data[ANIMALS[0]].stimuli)

    splits = generate_train_test_splits(
        num_stim=num_stim, num_splits=num_splits, train_frac=train_frac
    )

    results_list = []
    for s in splits:
        results = {}
        for animal_pair in itertools.permutations(ANIMALS, r=2):
            source_animal = animal_pair[0]
            target_animal = animal_pair[1]
            target_resp = brain_data[target_animal]
            source_resp = brain_data[source_animal]
            # cv is only on the train data
            if temporal:
                target_resp_train = target_resp.isel(stim_time=s["train"])
                source_resp_train = source_resp.isel(stim_time=s["train"])
            else:
                target_resp_train = target_resp.isel(stimuli=s["train"])
                source_resp_train = source_resp.isel(stimuli=s["train"])
            assert target_resp_train.ndim == 3
            assert source_resp_train.ndim == 3

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
                    num_bootstrap_iters=num_bootstrap_iters,
                    num_parallel_jobs=num_parallel_jobs,
                    splits=cv_splits,
                    metric=metric,
                )

        results_list.append(results)

    # fn = get_filename(
    #     map_name="ridgecv",
    #     brain_area=brain_area,
    #     interanimal=True,
    #     common_resp=True,
    #     num_splits=num_splits,
    #     num_cv_splits=num_cv_splits,
    #     train_frac=train_frac,
    #     trial_frac_lower_bound=trial_frac_lower_bound,
    #     additional_target_offset_sec=additional_target_offset_sec,
    #     num_bootstrap_iters=num_bootstrap_iters,
    #     metric=metric,
    #     enforce_finite_mean=enforce_finite_mean,
    #     time_mode=time_mode,
    #     temporal=temporal,
    #     collapse_temporal=True,
    #     trial_threshold=trial_threshold,
    #     start_offset_sec=start_offset_sec,
    #     downsample_rate=downsample_rate,
    # )
    # fn = os.path.join(NEURAL_FIT_CV_SEARCH_DIR, fn)
    # np.savez(fn, results_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-areas", type=str, default=None, required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--time-mode", type=str, default="target_gocue")
    parser.add_argument("--temporal", type=bool, default=False)
    parser.add_argument("--trial-threshold", type=int, default=360)
    parser.add_argument("--downsample-rate", type=int, default=None)
    parser.add_argument("--start-offset-sec", type=float, default=0.0)
    parser.add_argument("--trial-frac-lower-bound", type=float, default=0.5)
    parser.add_argument("--additional-target-offset-sec", type=float, default=0.2)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--num-bootstrap-iters", type=int, default=1000)
    parser.add_argument("--num-cv-splits", type=int, default=5)
    parser.add_argument("--num-parallel-jobs", type=int, default=1)
    parser.add_argument("--no-finite-mean-filt", type=bool, default=False)
    args = parser.parse_args()

    params = build_param_lookup(args)
    print(f"Num jobs: {len(list(params.keys()))}")
    curr_params = params[os.environ.get("SLURM_ARRAY_TASK_ID")]
    perform_cv(**curr_params)
