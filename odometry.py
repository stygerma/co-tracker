import os
from textwrap import wrap

import cv2
import numpy as np
import torch
from drone_movements._data import CameraPoseTracker
from drone_movements._plotting import plot_camera_poses
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy import typing as npt
from scipy.spatial.transform import Rotation as R
from sips.data import CameraPose

from cotracker.utils.config import Config
from cotracker.utils.io import IO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # makes tracking of larger videos possible as it avoids fragsmentation
)


class Odometry:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.io = IO(config)

        self.true_keypoints = self.get_true_keypoints()  # (num_frames, num_tracks, 2)
        self.source_video = self.get_source_video()  # (num_frames, H, W, C)

        self.pred_keypoints: npt.NDArray[np.float64] = np.array(
            []
        )  # (num_frames, num_tracks, 2)
        self.pred_visibilities: npt.NDArray[np.bool_] = np.array(
            []
        )  # (num_frames, H, W, C)

        # B is the batch size, which is 1 for synthetic data and 2 for real data
        # The first dimension is for the predicted and the second for the (true) synthetic keypoints
        self.odometry_transforms: npt.NDArray[np.float64] = np.array(
            []
        )  # (B, num_frames - 1, 2, 3)
        self.odometry_inliers: npt.NDArray[np.uint8] = np.array(
            []
        )  # (B, num_frames - 1, num_tracks)

        self.colormap = cm["cool"]

    # --------------------------------------------------------------------------
    # Load or generate "true" keypoints and the source video

    def get_true_keypoints(self) -> npt.NDArray[np.float64]:
        if not self.config.force_redo and self.io.get_true_keypoints_path().exists():
            true_keypoints = self.io.load_keypoints(pred=False)
        else:
            true_keypoints = self.generate_true_keypoints()
        assert true_keypoints.shape == (
            self.config.num_frames,
            self.config.num_tracks,
            2,
        )
        return true_keypoints

    def generate_true_keypoints(self) -> npt.NDArray[np.float64]:
        # Generate keypoints for synthetic data based on angle and distance
        if self.config.synthetic_data:
            height, width = self.config.image_shape
            true_keypoint_angles = self.config.true_keypoint_angles
            true_keypoint_distances = self.config.true_keypoint_distances

            assert len(true_keypoint_angles) == len(true_keypoint_distances)
            true_keypoints: npt.NDArray[np.float64] = np.full(
                (self.config.num_frames, len(true_keypoint_angles), 2), np.nan
            )
            angle_range = self.config.angular_fov // 2
            center = (width // 2, 0)  # Sonar origin at the top center

            for frame_num in range(self.config.num_frames):
                for track_num, (angle, distance) in enumerate(
                    zip(true_keypoint_angles, true_keypoint_distances)
                ):
                    angle = angle + frame_num * self.config.angular_step_size

                    # Check if the angle is within the range
                    if -angle_range >= angle >= angle_range:
                        # true_keypoints[frame_num].append([float("nan"), float("nan")])
                        continue

                    # Check if the distance is within the image bounds
                    if 0 > distance > height:
                        # true_keypoints[frame_num].append([float("nan"), float("nan")])
                        continue

                    # Convert angle to radians
                    angle_rad = np.radians(angle)

                    # Convert polar to Cartesian coordinates
                    x = center[0] + distance * np.sin(angle_rad)
                    y = center[1] + distance * np.cos(angle_rad)

                    true_keypoints[frame_num, track_num] = [x, y]
                    continue
            return true_keypoints

        # Get manually selected keypoints for real data
        manual_keypoints = self.config.manual_keypoints.get(self.config.file_name, None)
        if manual_keypoints is None:
            raise ValueError(
                f"No manually selected keypoints available for {self.config.file_name=}"
            )
        true_keypoints = np.full(
            (self.config.num_frames, len(manual_keypoints), 2), np.nan
        )
        true_keypoints[0] = manual_keypoints
        return true_keypoints

    def get_source_video(self) -> npt.NDArray[np.uint8]:
        source_video_path = self.io.get_source_video_path()
        if source_video_path.exists() and (
            not self.config.force_redo or not self.config.synthetic_data
        ):
            return self.io.load_source_video()[: self.config.num_frames]
        if not self.config.synthetic_data:
            raise FileNotFoundError(
                "Source video not found, needs to be provided for real data"
            )
        return self.generate_source_video()

    def _apply_keypoints_to_frames(
        self, video_frame: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        video_frames = []
        sigma = self.config.keypoint_sigma
        intensity = self.config.keypoint_intensity
        height, width = self.config.image_shape
        for frame_num in range(self.config.num_frames):
            frame = video_frame.copy()
            keypoints = self.true_keypoints[frame_num]
            for x, y in keypoints:
                if np.isnan(x) or np.isnan(y):
                    continue
                if not self.config.subpixel_accuracy:
                    # -1 fills the circle
                    cv2.circle(frame, (int(x), int(y)), sigma, intensity, -1)  # type: ignore[call-overload]
                    continue

                """Draws a Gaussian at subpixel (x, y) coordinates."""
                x0, y0 = int(x), int(y)  # Floor of coordinates

                # Create a fine-resolution kernel grid
                size = int(6 * sigma) | 1  # Ensure odd kernel size
                k = size // 2
                x_grid, y_grid = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

                # Compute Gaussian weights centered at the exact (subpixel) location
                gaussian = np.exp(
                    -((x_grid - (x - x0)) ** 2 + (y_grid - (y - y0)) ** 2)
                    / (2 * sigma**2)
                )
                # gaussian /= np.sum(gaussian)  # Normalize to maintain total intensity

                # Add the Gaussian to the image (handle boundaries)
                x_start, x_end = max(x0 - k, 0), min(x0 + k + 1, width)
                y_start, y_end = max(y0 - k, 0), min(y0 + k + 1, height)

                gx_start, gx_end = x_start - (x0 - k), x_end - (x0 - k)
                gy_start, gy_end = y_start - (y0 - k), y_end - (y0 - k)

                frame[y_start:y_end, x_start:x_end] += (
                    gaussian[gy_start:gy_end, gx_start:gx_end] * intensity
                ).astype(frame.dtype)
            video_frames.append(frame)
        return np.array(video_frames)

    def generate_source_video(self) -> npt.NDArray[np.uint8]:
        synthetic_sonar_frame = self.generate_synthetic_sonar_frame()
        fake_source_video = self._apply_keypoints_to_frames(synthetic_sonar_frame)

        # Duplicate the single channel to create a 3-channel image
        fake_source_video = np.stack([fake_source_video] * 3, axis=-1)
        return fake_source_video

    def generate_synthetic_sonar_frame(self) -> npt.NDArray[np.uint8]:
        # Define image resolution
        height, width = self.config.image_shape

        # Create a black background
        image = np.zeros((height, width), dtype=np.uint8)
        # Define sonar parameters
        center = (width // 2, 0)  # Sonar origin at the top center
        radius = height  # Frustum should reach the bottom

        # Compute the required opening angle to span the full width
        angle_range = self.config.angular_fov

        # Create a mask for the circular sector
        mask = np.zeros((height, width), dtype=np.uint8)

        # Define sector points manually to ensure correct positioning
        num_points = 500  # More points for a smooth arc
        angles = np.linspace(
            -np.radians(angle_range / 2), np.radians(angle_range / 2), num_points
        )
        x = (radius * np.sin(angles) + center[0]).astype(np.int32)
        y = (radius * np.cos(angles)).astype(np.int32)

        # Create the frustum polygon (sector shape)
        points = np.vstack((np.append(center[0], x), np.append(center[1], y))).T
        cv2.fillPoly(mask, [points.astype(np.int32)], (255,))

        # Apply the mask to the sonar frustum
        image[mask == 255] = 150

        # Add add_noise only within the masked region
        if self.config.add_sonar_noise:
            noise = np.random.normal(loc=0, scale=30, size=(height, width)).astype(
                np.int8
            )
            noise = cv2.bitwise_and(noise, noise, mask=mask).astype(
                np.int8
            )  # Apply mask to noise
            output_image = cv2.add(image, noise, dtype=cv2.CV_8U)  # .astype(np.uint8)
        else:
            output_image = image
        return output_image.astype(np.uint8)

    # --------------------------------------------------------------------------
    # Track the "true" keypoints with co-tracker

    def _compute_cotrack_keypoints(self) -> None:
        default_device = self.config.default_device
        if default_device == "cuda":
            torch.cuda.empty_cache()
        source_video = (
            torch.from_numpy(self.source_video).permute(0, 3, 1, 2)[None].float()
        ).to(default_device)  # (B, num_frames, 3, H, W)
        if self.config.offline_model:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        model = model.to(default_device)

        true_keypoints = torch.Tensor(self.true_keypoints).to(default_device)
        # The cotracker model expects the queries to have shape (B, num_tracks, 2)
        # with the batch number B being 1 for our case and the last dimension consisting
        # of the frame number and the x and y coordinates of the keypoints.
        true_keypoints = true_keypoints[0][None]
        true_keypoints = torch.cat(
            (
                torch.zeros(1, true_keypoints.shape[1], 1).to(default_device),
                true_keypoints,
            ),
            dim=2,
        )

        assert (
            true_keypoints.ndim == 3
            and true_keypoints.shape[2] == 3
            and true_keypoints.shape[0] == 1
        ), "Keypoint queries must have shape (B, num_tracks, 3)"

        if self.config.offline_model:
            pred_keypoints, pred_visibilities = model(
                source_video, queries=true_keypoints, backward_tracking=True
            )
        else:
            model(
                video_chunk=source_video,
                is_first_step=True,
                queries=true_keypoints,
            )

            for ind in range(0, source_video.shape[1] - model.step, model.step):
                pred_keypoints, pred_visibilities = model(
                    video_chunk=source_video[:, ind : ind + model.step * 2]
                )  # (B, num_frames, num_tracks, 2), (1, num_frames, num_tracks)
        self.pred_keypoints = pred_keypoints.cpu().numpy()[0]
        self.pred_visibilities = pred_visibilities.cpu().numpy()[0]
        if default_device == "cuda":
            torch.cuda.empty_cache()

    def cotrack_keypoints(self) -> None:
        print("Tracking keypoints with co-tracker")

        try:
            self._compute_cotrack_keypoints()
        except torch.OutOfMemoryError:
            print("Out of memory error occurred, switching to CPU")
            self.config.default_device = "cpu"
            self._compute_cotrack_keypoints()

        print("Finished keypoint tracking")
        return

    # --------------------------------------------------------------------------
    # Estimate the odometry for keypoint sequences

    def _apply_visibilities(
        self,
    ) -> npt.NDArray[np.float64]:
        visible_pred_keypoints = self.pred_keypoints.copy()

        visible_pred_keypoints[~self.pred_visibilities] = np.nan
        if self.config.needed_visibility_ratio > 0:
            frame_visibility_ratio = np.nanmean(self.pred_visibilities, axis=1)
            frame_visibility_too_low = (
                frame_visibility_ratio < self.config.needed_visibility_ratio
            )
            visible_pred_keypoints[frame_visibility_too_low] = np.nan
        return visible_pred_keypoints

    def estimate_odometry(self) -> None:
        print("Estimating odometry")
        if len(self.pred_keypoints) == 0:
            try:
                self.pred_keypoints = self.io.load_keypoints(pred=True)
                self.pred_visibilities = self.io.load_visibilities()
            except FileNotFoundError:
                self.cotrack_keypoints()
        assert (
            self.pred_keypoints.shape == self.true_keypoints.shape
            and self.pred_keypoints.shape[1] == self.config.num_tracks
            and self.pred_visibilities.shape[:2] == self.pred_keypoints.shape[:2]
        )
        odometry_transforms = np.zeros((self.config.num_frames - 1, 2, 3))
        odometry_inliers = np.zeros(
            (self.config.num_frames - 1, self.config.num_tracks), dtype=np.uint8
        )
        if self.config.check_visibilities:
            pred_keypoints = self._apply_visibilities()
        else:
            pred_keypoints = self.pred_keypoints
        if self.config.synthetic_data:
            self.odometry_transforms = np.stack([odometry_transforms] * 2, axis=0)
            self.odometry_inliers = np.stack([odometry_inliers] * 2, axis=0)
            all_keypoints = np.stack([pred_keypoints, self.true_keypoints], axis=0)
        else:
            self.odometry_transforms = np.stack([odometry_transforms] * 1, axis=0)
            self.odometry_inliers = np.stack([odometry_inliers] * 1, axis=0)
            all_keypoints = np.stack([pred_keypoints], axis=0)
        for kps, keypoints in enumerate(all_keypoints):
            assert keypoints.ndim == 3
            for frame_num in range(self.config.num_frames - 1):
                src = keypoints[frame_num]
                dst = keypoints[frame_num + 1]

                transforms, inliers = self._calculate_transform(src.copy(), dst.copy())
                self.odometry_transforms[kps, frame_num] = transforms
                self.odometry_inliers[kps, frame_num] = inliers
        print("Finished odometry estimation")
        return

    def _calculate_transform(
        self, src: npt.NDArray[np.float64], dst: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        src_centered = src
        dst_centered = dst

        src_centered[:, 0] = src_centered[:, 0] - (1116 // 2)
        dst_centered[:, 0] = dst_centered[:, 0] - (1116 // 2)
        T, inl = cv2.estimateAffinePartial2D(
            src_centered,
            dst_centered,
            method=cv2.RANSAC,
            ransacReprojThreshold=5,
        )
        if T is None:
            T = np.full((2, 3), np.nan)
        return T.astype(np.float64), inl.astype(np.uint8).squeeze()

    # --------------------------------------------------------------------------
    # Odometry analysis with plots and metrics

    def _xs(
        self, T: npt.NDArray[np.float64], meters: bool = True
    ) -> npt.NDArray[np.float64]:
        assert (T.ndim == 2 or T.ndim == 3) and T.shape[-2:] == (2, 3), (
            "Transformation(s) must have shape (2, 3) or (num_frames, 2, 3)"
        )
        if T.ndim == 2:
            if meters:
                return T[0, 2] * self.config.range_resolution
            return T[0, 2]
        if meters:
            return T[:, 0, 2] * self.config.range_resolution
        return T[:, 0, 2]

    def _ys(
        self, T: npt.NDArray[np.float64], meters: bool = True
    ) -> npt.NDArray[np.float64]:
        assert (T.ndim == 2 or T.ndim == 3) and T.shape[-2:] == (2, 3), (
            "Transformation(s) must have shape (2, 3) or (num_frames, 2, 3)"
        )
        if T.ndim == 2:
            if meters:
                return T[1, 2] * self.config.range_resolution
            return T[1, 2]
        if meters:
            return T[:, 1, 2] * self.config.range_resolution
        return T[:, 1, 2]

    def _yaws(
        self, T: npt.NDArray[np.float64], degrees: bool = False
    ) -> float | npt.NDArray[np.float64]:
        assert (T.ndim == 2 or T.ndim == 3) and T.shape[-2:] == (2, 3), (
            "Transformation(s) must have shape (2, 3) or (num_frames, 2, 3)"
        )
        if T.ndim == 2:
            yaw = np.arctan2(T[1, 0], T[0, 0])
            if degrees:
                return np.degrees(yaw)
            return yaw
        else:
            yaws = np.arctan2(T[:, 1, 0], T[:, 0, 0])
            if degrees:
                return np.degrees(yaws)
            return yaws

    def plot_keypoint_trajectories(self, display_track_numbers: bool = False) -> Axes:
        height, width = self.config.image_shape
        _, ax = plt.subplots()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        origin = np.array([width // 2, 0])

        # Store easy labels for the legend
        all_labels: list[Line2D | PathCollection] = []

        # Plot vertical line signifying the sonar center
        all_labels.append(
            ax.axvline(
                x=origin[0],
                color="black",
                linestyle="--",
                label="Sonar center",
                alpha=0.5,
            )
        )

        num_tracks = self.config.num_tracks
        colors = self.colormap(np.linspace(0, 1, num_tracks + 2))
        label_for_empty_points_is_set = False

        for track in range(num_tracks):
            # Plot true tracks with solid lines
            ax.plot(
                self.true_keypoints[:, track, 0],
                self.true_keypoints[:, track, 1],
                marker="o",
                linestyle="-",
                color=colors[track],
                label=f"True Keypoints of Track {track}" if track == 0 else "",
            )

            nan_mask = np.logical_or(
                np.isnan(self.true_keypoints[:, track, 0]),
                np.isnan(self.true_keypoints[:, track, 1]),
            )

            face_colors = np.array([colors[track]] * self.pred_keypoints.shape[0])
            face_colors[~self.pred_visibilities[:, track]] = [
                1.0,
                1.0,
                1.0,
                1.0,
            ]  # Fill invisible points with white
            if not label_for_empty_points_is_set and nan_mask.sum() > 0:
                label_for_empty_points_is_set = True

            # Plot the predicted keypoints with circles that are only filled if
            # the corresponding true keypoint exists otherwise they are empty
            ax.scatter(
                self.pred_keypoints[:, track, 0],
                self.pred_keypoints[:, track, 1],
                marker="o",
                color=colors[track],
                # label=labels,
                edgecolor=colors[track],
                facecolor=face_colors,
            )

            # Connect the predicted keypoints with dashed lines
            ax.plot(
                [
                    self.pred_keypoints[:-1, track, 0],
                    self.pred_keypoints[1:, track, 0],
                ],
                [
                    self.pred_keypoints[:-1, track, 1],
                    self.pred_keypoints[1:, track, 1],
                ],
                linestyle="-.",
                color=colors[track],
            )

            ax.plot(
                [
                    self.true_keypoints[:, track, 0],
                    self.pred_keypoints[:, track, 0],
                ],
                [
                    self.true_keypoints[:, track, 1],
                    self.pred_keypoints[:, track, 1],
                ],
                linestyle="-",
                color="red",
            )
        all_labels.append(
            ax.scatter(
                origin[0],
                origin[1],
                marker="^",
                color=colors[num_tracks],
                label="Sonar origin",
            )
        )

        all_labels.append(
            ax.scatter(
                self.true_keypoints[0, :, 0],
                self.true_keypoints[0, :, 1],
                marker=">",
                color=colors[num_tracks + 1],
                zorder=2,
                label="True starting points",
            )
        )

        if display_track_numbers:
            for i, (x, y) in enumerate(
                zip(self.true_keypoints[0, :, 0], self.true_keypoints[0, :, 1])
            ):
                ax.text(x, y, str(i))

        custom_lines = [
            Line2D([0], [0], color="red", lw=2, label="Errors"),
            Line2D(
                [0],
                [0],
                color=colors[0],
                linestyle="-",
                lw=2,
                marker="o",
                markersize=5,
                label='"True" Keypoints of Track 0',
            ),
            Line2D(
                [0],
                [0],
                color=colors[0],
                linestyle="-.",
                lw=2,
                marker="o",
                markersize=5,
                markeredgecolor=colors[0],
                markerfacecolor=colors[0],
                label="Visible predicted Keypoints of Track 0",
            ),
            Line2D(
                [0],
                [0],
                color=colors[0],
                linestyle="-.",
                lw=2,
                marker="o",
                markersize=5,
                markeredgecolor=colors[0],
                markerfacecolor="white",
                label="Invisible predicted Keypoints of Track 0",
            ),
        ]

        ax.set_title(
            "\n".join(
                wrap(
                    f"True (synthetic) and co-tracker predicted keypoint locations over time ({self.io.get_full_file_name(with_model=True)})",
                    60,
                )
            )
        )
        ax.legend(handles=all_labels + custom_lines)
        return ax

    def plot_odometry_transformations(
        self, meters: bool = True, degrees: bool = True
    ) -> tuple[Axes, Axes]:
        if self.odometry_transforms.shape[0] == 2:
            _, true_ax = plt.subplots()
        else:
            true_ax = None

        _, pred_ax = plt.subplots()

        # Get colors
        colors = self.colormap(np.linspace(0, 1, 3))

        for ax, transforms, name in zip(
            [pred_ax, true_ax],
            [*self.odometry_transforms],
            ["co-tracker predicted", "true (synthetic)"],
        ):
            assert transforms.ndim == 3
            if ax is None:
                continue
            # Make sure that both potential plots have the same x-axis limits
            # to make the comparison easier. The offsets are chosen based on
            # default offsets of matplotlib.
            ax.set_xlim(-0.65, self.odometry_transforms.shape[1] - 0.35)
            xs = ax.plot(
                [self._xs(transform, meters=meters) for transform in transforms],
                label="xs",
                marker=".",
                linestyle="-.",
                c=colors[0],
            )
            ys = ax.plot(
                [self._ys(transform, meters=meters) for transform in transforms],
                label="ys",
                marker=".",
                linestyle="--",
                c=colors[1],
            )

            ax_twin = ax.twinx()
            yaws = ax_twin.plot(
                self._yaws(transforms, degrees=degrees),
                marker=".",
                linestyle="-",
                label="yaws",
                c=colors[2],
            )
            # The y-axis dimensions were displayed in unintuitve scientific notation
            # without this (1e-6-5) which this line fixes.
            ax_twin.get_yaxis().get_major_formatter().set_useOffset(False)

            lns = xs + ys + yaws
            labs = [line.get_label() for line in lns]
            ax.legend(lns, labs)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
            translation_ylabel = (
                "Translation (meters)" if meters else "Translation (pixels)"
            )
            ax.set_ylabel(translation_ylabel)
            rotation_ylabel = "Yaw (radians)" if not degrees else "Yaw (degrees)"
            ax_twin.set_ylabel(rotation_ylabel, color=colors[2])
            ax_twin.tick_params(axis="y", labelcolor=colors[2])
            ax.set_title(
                "\n".join(
                    wrap(
                        f"Odometry-estimated transformations for {name} keypoints ({self.io.get_full_file_name(with_model=True)})",
                        60,
                    )
                )
            )
        return true_ax, pred_ax

    def calculate_poses_from_transforms(
        self,
        Ts: npt.NDArray[np.float64],
        tracker: CameraPoseTracker,
        degrees: bool = False,
        meters: bool = True,
    ) -> None:
        t, _, _ = Ts.shape

        for i in range(t):
            if np.isnan(Ts[i]).sum():
                continue
            transform_yaw = self._yaws(Ts[i], degrees=degrees)
            transform_quat = R.from_euler("z", transform_yaw, degrees=degrees).as_quat()
            if len(tracker) - 1 < i:
                continue
            old_quat = tracker[i][0].rotation
            old_pos = tracker[i][0].position
            updated_quat = (
                R.from_quat(old_quat) * R.from_quat(transform_quat)
            ).as_quat()
            x_offset = self._xs(Ts[i], meters=meters).item()
            y_offset = self._ys(Ts[i], meters=meters).item()
            updated_pos = [
                old_pos[0].item() + x_offset,
                old_pos[1].item() + y_offset,
                old_pos[2].item(),
            ]

            tracker.move_abs(CameraPose(updated_pos, updated_quat))
        return

    def _get_init_pose(self) -> CameraPose:
        if self.config.file_name in self.config.init_pose:
            return self.config.init_pose[self.config.file_name]
        return self.config.init_pose["default"]

    def plot_odometry_estimated_poses(self, meters: bool = True) -> list[Axes]:
        axes: list[Axes] = []
        custom_cm = cm["gist_rainbow"]
        colors = custom_cm(np.linspace(0, 1, self.odometry_transforms.shape[1]))
        for transforms, name in zip(
            [*self.odometry_transforms],
            ["co-tracker predicted", "true (synthetic)"],
        ):
            assert transforms.ndim == 3

            pose_tracker = CameraPoseTracker(self._get_init_pose())
            self.calculate_poses_from_transforms(
                transforms, pose_tracker, meters=meters
            )
            poses_ax = plot_camera_poses(pose_tracker, show=False, color=colors)
            poses_ax.scatter(
                pose_tracker[0][0].position[0],
                pose_tracker[0][0].position[1],
                pose_tracker[0][0].position[2],
                color="lime",
                label="Origin",
            )
            axes.append(poses_ax)
            # Connect subsequent poses with lines
            for i in range(len(pose_tracker) - 1):
                positions = np.vstack(
                    (pose_tracker[i][0].position, pose_tracker[i + 1][0].position)
                )
                poses_ax.plot(
                    *positions.T,
                    color="black",
                    alpha=0.5,
                    label="Trajectory" if i == 0 else "",
                )
            poses_ax.legend()
            distance_unit = "meters" if meters else "pixels"
            poses_ax.set_title(
                "\n".join(
                    wrap(
                        f"Predicted pose trajectory based on odometry of {len(pose_tracker)} {name} keypoints in {distance_unit} ({self.io.get_full_file_name(with_model=True)})",
                        60,
                    )
                )
            )
        return axes

    def analyze_odometry(self) -> None:
        if len(self.pred_keypoints) == 0:
            try:
                self.pred_keypoints = self.io.load_keypoints(pred=True)
                self.pred_visibilities = self.io.load_visibilities()
            except FileNotFoundError:
                self.cotrack_keypoints()

        if len(self.odometry_transforms) == 0:
            try:
                self.odometry_transforms = self.io.load_odometry_transforms()
                self.odometry_inliers = self.io.load_odometry_inliers()
            except FileNotFoundError:
                self.estimate_odometry()

        _ = self.plot_keypoint_trajectories()
        _ = self.plot_odometry_transformations()
        _ = self.plot_odometry_estimated_poses()

        plt.show()
        return

    def save_data(self) -> None:
        self.io.store_keypoints(self.true_keypoints, pred=False)
        self.io.store_synthetic_video(self.source_video)

        # cotrack_keypoints() has been called
        if len(self.pred_keypoints) > 0:
            self.io.store_keypoints(self.pred_keypoints, pred=True)
            self.io.store_visibilities(self.pred_visibilities)
            self.io.store_cotracker_video(
                self.source_video, self.pred_keypoints, self.pred_visibilities
            )

        # analyze_odometry() has been called
        if len(self.odometry_transforms) > 0:
            self.io.store_odometry_transforms(self.odometry_transforms)
            self.io.store_odometry_inliers(self.odometry_inliers)
        print("Data saved successfully")
        return


def main() -> None:
    config = Config()
    odo = Odometry(config)
    odo.cotrack_keypoints()
    odo.estimate_odometry()
    # odo.analyze_odometry()
    odo.save_data()
    return


if __name__ == "__main__":
    main()
