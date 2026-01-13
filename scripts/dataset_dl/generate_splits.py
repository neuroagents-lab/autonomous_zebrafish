import argparse
import numpy as np
import os
from zfa.core.default_dirs import MVK_RAW_DIR, MVK_FRAME_DIR
from ptutils.core.video_utils import assign_videos_to_sets

if __name__ == "__main__":

    # generate the splits
    parser = argparse.ArgumentParser(
        description="Assign videos to train, validation, and test sets."
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

    train_videos, val_videos, test_videos = assign_videos_to_sets(
        base_dir=MVK_RAW_DIR,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    # Ensure the processed directory exists
    processed_dir = MVK_FRAME_DIR
    os.makedirs(processed_dir, exist_ok=True)

    # Save the filenames to an npz file
    np.savez(
        os.path.join(
            processed_dir,
            f"dataset_splits_{args.train_frac}_{args.val_frac}_{args.test_frac}.npz",
        ),
        train=train_videos,
        val=val_videos,
        test=test_videos,
    )

    # For demonstration, print out the counts of videos in each set
    print(f"Train videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")
