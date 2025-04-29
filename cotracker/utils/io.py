import json
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
from numpy import typing as npt

from cotracker.utils.config import Config
from cotracker.utils.visualizer import Visualizer, read_video_from_path


class IO:
    def __init__(
        self,
        config: Config,
        *,
        data_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.data_dir = Path(data_dir or config.data_dir)

    # --------------------------------------------------------------------------
    # File names
    # --------------------------------------------------------------------------

    def get_full_file_name(
        self, with_model: bool = False, with_visibility: bool = False
    ) -> str:
        file_name = self.config.file_name
        if with_model:
            if self.config.offline_model:
                file_name = f"{self.config.model_name}_{self.config.offline_model_backward_tracking}_{file_name}"
            else:
                file_name = f"{self.config.model_name}_{file_name}"
        if with_visibility:
            if self.config.check_visibilities:
                file_name = f"{file_name}_{self.config.check_visibilities}_{self.config.needed_visibility_ratio}"
            else:
                file_name = f"{file_name}_{self.config.check_visibilities}"
        return file_name

    def get_video_name(self, with_model: bool = False) -> str:
        return f"{self.get_full_file_name(with_model)}.mp4"

    def get_csv_name(
        self, with_model: bool = False, with_visibility: bool = False
    ) -> str:
        return f"{self.get_full_file_name(with_model, with_visibility)}.csv"

    # --------------------------------------------------------------------------
    # Directories and paths getters
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Video directories and paths

    def get_source_video_dir(self) -> Path:
        return self.data_dir / "source_videos"

    def get_source_video_path(self) -> Path:
        return self.get_source_video_dir() / self.get_video_name(with_model=False)

    def get_saved_video_dir(self) -> Path:
        return self.data_dir / "saved_videos"

    # --------------------------------------------------------------------------
    # Keypoint visibilities directories and paths

    def get_visibilities_dir(self) -> Path:
        return self.get_pred_keypoints_dir() / "visibilities"

    def get_visibilities_path(self) -> Path:
        return self.get_visibilities_dir() / self.get_csv_name(with_model=True)

    # --------------------------------------------------------------------------
    # Keypoints directories and paths

    def get_true_keypoints_dir(self) -> Path:
        return self.data_dir / "keypoints/true"

    def get_true_keypoints_path(self) -> Path:
        return self.get_true_keypoints_dir() / self.get_csv_name(with_model=False)

    def get_pred_keypoints_dir(self) -> Path:
        return self.data_dir / "keypoints/pred"

    def get_pred_keypoints_path(self) -> Path:
        return self.get_pred_keypoints_dir() / self.get_csv_name(with_model=True)

    def get_keypoints_path(self, pred: bool = False) -> Path:
        if pred:
            return self.get_pred_keypoints_path()
        return self.get_true_keypoints_path()

    # --------------------------------------------------------------------------
    # Odometry transforms directories and paths

    def get_true_odometry_transforms_dir(self) -> Path:
        return self.data_dir / "odometry/transforms/true"

    def get_true_odometry_transforms_path(self) -> Path:
        return self.get_true_odometry_transforms_dir() / self.get_csv_name(
            with_model=True, with_visibility=True
        )

    def get_pred_odometry_transforms_dir(self) -> Path:
        return self.data_dir / "odometry/transforms/pred"

    def get_pred_odometry_transforms_path(self) -> Path:
        return self.get_pred_odometry_transforms_dir() / self.get_csv_name(
            with_model=True, with_visibility=True
        )

    def get_odometry_transforms_path(self, pred: bool = True) -> Path:
        if pred:
            return self.get_pred_odometry_transforms_path()
        return self.get_true_odometry_transforms_path()

    # --------------------------------------------------------------------------
    # Odometry inliers directories and paths

    def get_true_odometry_inliers_dir(self) -> Path:
        return self.data_dir / "odometry/inliers/true"

    def get_true_odometry_inliers_path(self) -> Path:
        return self.get_true_odometry_inliers_dir() / self.get_csv_name(
            with_model=True, with_visibility=True
        )

    def get_pred_odometry_inliers_dir(self) -> Path:
        return self.data_dir / "odometry/inliers/pred"

    def get_pred_odometry_inliers_path(self) -> Path:
        return self.get_pred_odometry_inliers_dir() / self.get_csv_name(
            with_model=True, with_visibility=True
        )

    def get_odometry_inliers_path(self, pred: bool = True) -> Path:
        if pred:
            return self.get_pred_odometry_inliers_path()
        return self.get_true_odometry_inliers_path()

    # --------------------------------------------------------------------------
    # Load and store functions
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Keypoints load and store

    def load_keypoints(self, pred: bool = False) -> npt.NDArray[np.float64]:
        keypoints = pd.read_csv(
            self.get_keypoints_path(pred=pred), header=None
        ).to_numpy()

        keypoints = keypoints.reshape(
            keypoints.shape[0], -1, 2
        )  # (num_frames, num_tracks, 2)
        assert keypoints.shape == (self.config.num_frames, self.config.num_tracks, 2)
        return keypoints

    def store_keypoints(
        self, keypoints: npt.NDArray[np.float64], pred: bool = False
    ) -> None:
        if pred:
            file_path = self.get_pred_keypoints_path()
        else:
            file_path = self.get_true_keypoints_path()
        if keypoints.ndim == 2:
            pd.DataFrame(keypoints).to_csv(file_path, index=False, header=False)
            return
        pd.DataFrame(keypoints.reshape(self.config.num_frames, -1)).to_csv(
            file_path, index=False, header=False
        )
        return

    # --------------------------------------------------------------------------
    # Visibilities load and store

    def load_visibilities(self) -> npt.NDArray[np.bool_]:
        visibilities = pd.read_csv(self.get_visibilities_path(), header=None).to_numpy()
        assert visibilities.shape == (self.config.num_frames, self.config.num_tracks)
        return visibilities

    def store_visibilities(self, visibilities: npt.NDArray[np.bool_]) -> None:
        pd.DataFrame(visibilities).to_csv(
            self.get_visibilities_path(), index=False, header=False
        )
        return

    # --------------------------------------------------------------------------
    # Videos load and store

    def load_source_video(self) -> npt.NDArray[np.uint8]:
        return read_video_from_path(self.get_source_video_path())

    def store_synthetic_video(self, video_frames: npt.NDArray[np.uint8]) -> None:
        video_path = self.get_source_video_path()
        video_writer = imageio.get_writer(video_path, fps=5, macro_block_size=1)
        for frame in video_frames:
            video_writer.append_data(frame)
        video_writer.close()

    def store_cotracker_video(
        self,
        video: npt.NDArray[np.uint8],
        pred_keypoints: npt.NDArray[np.float64],
        pred_visibilities: npt.NDArray[np.bool_],
        tracks_leave_trace: int = -1,
    ) -> None:
        vis = Visualizer(
            save_dir=str(self.get_saved_video_dir()),
            # pad_value=120,
            linewidth=3,
            fps=5,
            tracks_leave_trace=tracks_leave_trace,
        )
        vis.visualize(
            torch.tensor(video).permute(0, 3, 1, 2).unsqueeze(0),
            torch.tensor(pred_keypoints).unsqueeze(0),
            torch.tensor(pred_visibilities).unsqueeze(0),
            query_frame=0,
            filename=self.get_video_name(with_model=True).split(".mp4")[
                0
            ],  # The visualizer adds the file extension
        )

    # --------------------------------------------------------------------------
    # Odometry transforms load and store

    def store_odometry_transforms(self, transforms: npt.NDArray[np.float64]) -> None:
        assert transforms.ndim == 4
        preds = [True, False]
        for transform, pred in zip(transforms, preds):
            assert transform.shape == (self.config.num_frames - 1, 2, 3)
            pd.DataFrame(transform.reshape(-1, 6)).to_csv(
                self.get_odometry_transforms_path(pred=pred), index=False, header=False
            )
        return

    def load_odometry_transforms(self) -> npt.NDArray[np.float64]:
        pred_odometry_transforms = (
            pd.read_csv(self.get_odometry_transforms_path(pred=True), header=None)
            .to_numpy()
            .reshape(-1, 2, 3)
        )

        if self.get_odometry_transforms_path(pred=False).exists():
            true_odometry_transforms = (
                pd.read_csv(self.get_odometry_transforms_path(pred=False), header=None)
                .to_numpy()
                .reshape(-1, 2, 3)
            )
            odometry_transforms = np.array(
                [pred_odometry_transforms, true_odometry_transforms]
            )
            assert odometry_transforms.shape == (2, self.config.num_frames - 1, 2, 3)
            return odometry_transforms
        odometry_transforms = np.array([pred_odometry_transforms])
        assert odometry_transforms.shape == (1, self.config.num_frames - 1, 2, 3)
        return odometry_transforms

    # --------------------------------------------------------------------------
    # Odometry inliers load and store

    def store_odometry_inliers(self, inliers: npt.NDArray[np.uint8]) -> None:
        assert inliers.ndim == 3
        preds = [True, False]
        for inlier, pred in zip(inliers, preds):
            assert inlier.shape == (self.config.num_frames - 1, self.config.num_tracks)
            pd.DataFrame(inlier.reshape(-1, self.config.num_tracks)).to_csv(
                self.get_odometry_inliers_path(pred=pred), index=False, header=False
            )
        return

    def load_odometry_inliers(self) -> npt.NDArray[np.uint8]:
        pred_odometry_inliers = pd.read_csv(
            self.get_odometry_inliers_path(pred=True), header=None
        ).to_numpy()

        if self.get_odometry_inliers_path(pred=False).exists():
            true_odometry_inliers = pd.read_csv(
                self.get_odometry_inliers_path(pred=False), header=None
            ).to_numpy()
            odometry_inliers = np.array([pred_odometry_inliers, true_odometry_inliers])
            assert odometry_inliers.shape == (
                2,
                self.config.num_frames - 1,
                self.config.num_tracks,
            )
            return odometry_inliers
        odometry_inliers = np.array([pred_odometry_inliers])
        assert odometry_inliers.shape == (
            1,
            self.config.num_frames - 1,
            self.config.num_tracks,
        )
        return odometry_inliers

    def load_true_poses(self) -> npt.NDArray[np.float64]:
        if self.config.synthetic_data or self.config.file_name not in [
            "sonarImageStructure",
            "PolarSonarImageStructure",
        ]:
            raise ValueError(
                "True poses are only available for the sonarImageStructure and PolarSonarImageStructure"
            )
        with open(
            "co-tracker/data/source_data/poses/interpolated_sonarImageStructure.json"
        ) as f:
            true_poses_with_stamp = json.load(f)
        true_poses = list(true_poses_with_stamp.values())
        return np.array(true_poses)
