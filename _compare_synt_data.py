import importlib
import itertools

import pandas as pd
from tqdm import tqdm

from cotracker.utils.config import Config

odometry = importlib.import_module("co-tracker.odometry")

params = {
    "offline_model": [True],
    "offline_model_backward_tracking": [True, False],
    "subpixel_accuracy": [True, False],
    "add_sonar_noise": [False],
    # "check_visibilities": [True, False],
    # "needed_visibility_ratio": [0.0, 0.2, 0.8],
}
combinations = list(itertools.product(*params.values()))
for combination in tqdm(combinations):
    (
        offline_model,
        offline_model_backward_tracking,
        subpixel_accuracy,
        add_sonar_noise,
        # check_visibilities,
        # needed_visibility_ratio,
    ) = combination
    config = Config(
        offline_model=offline_model,
        offline_model_backward_tracking=offline_model_backward_tracking,
        subpixel_accuracy=subpixel_accuracy,
        add_sonar_noise=add_sonar_noise,
        # check_visibilities=check_visibilities,
        # needed_visibility_ratio=needed_visibility_ratio,
    )

    odo = odometry.Odometry(config)
    odo.cotrack_keypoints()
    ax = odo.plot_keypoint_trajectories()
    ax.set_title(f"{combination}")

df = pd.read_csv("co-tracker/synt_data_comparison.csv")
df = df.sort_values(by=["MSE"])
