import numpy as np
import os
import argparse

from zfa.core.default_dirs import MVK_RAW_DIR, MVK_FRAME_DIR
from ptutils.core.video_utils import run_queue, video_to_rgb

"""
Given individual video files (mp4, webm) on disk, creates a folder for
every video file and saves the video's RGB frames as jpeg files in that
folder.

Uses multithreading to extract frames faster.
"""


# we keep these functions here since the patterning is custom to the application
def strip_prefix_from_filenames(video_filenames, prefix):
    # Initialize a new list to store filenames with the prefix removed
    new_filenames = []

    # Iterate through each filename in the list
    for filename in video_filenames:
        assert prefix.endswith("/")
        # Check if the filename starts with the specified prefix
        assert filename.startswith(prefix)
        new_filenames.append(filename[len(prefix) :])

    # Return the new list of filenames
    return new_filenames


def process_videofile(
    video_filename,
    video_in_path,
    rgb_out_path,
    shortest_side_size,
    file_extension: str = ".mp4",
):
    filepath = os.path.join(video_in_path, video_filename)
    # remove .mp4 and /, so it becomes "X_01" from "X/01.mp4"
    video_filename = video_filename.replace("/", "_").replace(file_extension, "")
    out_dir = os.path.join(rgb_out_path, video_filename)
    video_to_rgb(
        video_filename=filepath,
        out_dir=out_dir,
        shortest_side_size=shortest_side_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script to extract the jpgs from videos."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        required=True,
        choices=["train", "val", "test"],
        help="Which video mode to extract to.",
    )
    parser.add_argument(
        "--shortest-side-size",
        type=int,
        default=480,
        help="Set the shortest side of the extracted jpeg. Sets it to 480 by default.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=20,
        help="How many threads to use during extraction.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.9,
        help="Proportion of the dataset to include in the train split",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Proportion of the dataset to include in the validation split",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.05,
        help="Proportion of the dataset to include in the test split",
    )
    args = parser.parse_args()

    # the path to the folder which contains all video files (mp4, webm, or other)
    video_in_path = MVK_RAW_DIR
    # the path where we write out rgbs
    rgb_out_path = os.path.join(MVK_FRAME_DIR, args.mode)
    # note: need to run generate_splits.py first to generate the splits
    splits_file_path = os.path.join(
        MVK_FRAME_DIR,
        f"dataset_splits_{args.train_frac}_{args.val_frac}_{args.test_frac}.npz",
    )
    # get the video filenames for that particular split
    video_filenames = np.load(splits_file_path)[args.mode].tolist()
    stripped_video_filenames = strip_prefix_from_filenames(
        video_filenames=video_filenames, prefix=video_in_path
    )

    # format will be rgb_out_path/video_filename/frameXXXX.png, by default
    run_queue(
        args=args,
        process_video_func=process_videofile,
        video_in_path=video_in_path,
        rgb_out_path=rgb_out_path,
        shortest_side_size=args.shortest_side_size,
        video_filenames=stripped_video_filenames,
    )
