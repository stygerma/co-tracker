from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt
from sips.data import CameraPose


@dataclass
class Config:
    data_dir: str = "co-tracker/data"

    image_shape: tuple[int, int] = (
        522,
        1024,
    )  # height and width in cartesian coordinates

    angular_fov: int = 130
    max_range: float = 5.0
    range_resolution: float = 0.009571

    file_name: str = "sonarImageStructure"

    rng = np.random.default_rng(42)

    num_frames: int = 400
    _num_tracks: int = field(repr=False, init=False)
    synthetic_data: bool = False
    random_synthetic_data: bool = True
    num_synthetic_tracks: int = 100
    offline_model: bool = False
    offline_model_backward_tracking: bool = False
    force_redo: bool = True

    compare_cotracker_models: bool = False

    default_device: str = "cuda"

    # Synthetic keypoint parameters
    subpixel_accuracy: bool = True
    add_sonar_noise: bool = False
    true_keypoint_distances: list[float] = field(
        default_factory=lambda: [300, 320, 330, 540, 560, 400, 290, 500, 200]  #
    )
    true_keypoint_angles: list[float] = field(
        default_factory=lambda: [-35, -35, -35, -45, -45, -86, 19, -45, -64]  #
    )
    true_keypoint_starting_frames: list[int] = field(
        default_factory=lambda: [0, 0, 0, 0, 5, 0, 0, 0, 0]  #
    )
    angular_step_size: int = 5
    keypoint_sigma: int = 2
    keypoint_intensity: int = 255

    check_visibilities: bool = True
    needed_visibility_ratio: float = (
        0.0  # 0.0 keeps all frames and only applies individual exclusion
    )

    plot_all_datapoints: bool = False

    brightness_threshold: int = 230

    # Manually selected keypoints for real data
    manual_keypoints: dict[str, npt.NDArray[np.float64]] = field(
        default_factory=lambda: {
            "sonarImageStructure": np.array(
                [
                    [0.0, 434.2, 338.1],
                    [0.0, 439.9, 164.3],
                    [0.0, 507.9, 181.5],
                    [0.0, 669.4, 378.1],
                    [0.0, 582.9, 191.5],
                    [0.0, 610.8, 175.1],
                    [4.0, 539.0, 405.1],
                ]
            ),
            "PolarSonarImageStructure": np.array(
                [
                    [0.0, 204.0, 347.3],
                    [0.0, 161.1, 180.0],
                    [0.0, 250.8, 181.5],
                    [0.0, 346.6, 410.8],
                    [0.0, 337.5, 204.7],
                    [0.0, 373.8, 202.0],
                    [4.0, 271.3, 406.0],
                ]
            ),
            "rerecordLimmatCartesian": np.array(
                [
                    [0.0, 629.1, 182.4],
                    [0.0, 477.2, 215.2],
                    [0.0, 481.8, 569.7],
                    [0.0, 541.1, 226.8],
                    [0.0, 732.7, 327.4],
                    [0.0, 573.0, 210.5],
                    [0.0, 474.0, 192.6],
                ]
            ),
            "rerecordLimmatCartesianPoleApproach": np.array(
                [
                    [0.0, 135.1, 359.3],
                    [0.0, 678.2, 346.1],
                    [0.0, 689.9, 332.8],
                    [0.0, 713.3, 344.5],
                    [0.0, 781.1, 333.6],
                    [0.0, 619.8, 168.4],
                ],
                dtype=np.float64,
            ),
        }
    )

    init_pose: dict[str, CameraPose] = field(
        default_factory=lambda: {
            "default": CameraPose.neutral_pose(),
            "interpolated_sonarImageStructure": CameraPose(  # interpolated data
                [-4.904834105602733, 2.577402348695038, -3.4409040879928843],
                [
                    -0.018202238128860054,
                    0.0037275872930560695,
                    0.5298373970162384,
                    0.8478956208875751,
                ],
            ),
            "sonarImageStructure": CameraPose(  # raw data
                [-4.9045622151498804, 2.577873106620971, -3.4396288313875956],
                [
                    -0.018066443533008323,
                    0.003689051080770826,
                    0.529912156222455,
                    0.8478520514853168,
                ],
            ),
            "PolarSonarImageStructure": CameraPose(  # raw data
                [-4.9045622151498804, 2.577873106620971, -3.4396288313875956],
                [
                    -0.018066443533008323,
                    0.003689051080770826,
                    0.529912156222455,
                    0.8478520514853168,
                ],
            ),
        }
    )

    def __post_init__(self) -> None:
        if self.synthetic_data:
            self._num_tracks = len(self.true_keypoint_distances)
            self.file_name = "fake_sonar"
            if self.subpixel_accuracy:
                self.file_name += "_subpixel"
            if self.add_sonar_noise:
                self.file_name += "_noisy"
            if self.random_synthetic_data:
                self.true_keypoint_angles = self.rng.integers(
                    low=-self.angular_fov // 2 - 50,
                    high=self.angular_fov // 2,
                    size=self.num_synthetic_tracks,
                )
                self.true_keypoint_distances = self.rng.integers(
                    low=0, high=self.image_shape[0], size=self.num_synthetic_tracks
                )
                self.true_keypoint_starting_frames = np.zeros(self.num_synthetic_tracks)
        else:
            self._num_tracks = len(self.manual_keypoints[self.file_name])

    @property
    def num_tracks(self) -> int:
        return self._num_tracks

    @num_tracks.setter
    def num_tracks(self, value: int) -> None:
        self._num_tracks = value

    @property
    def model_name(self) -> str:
        if self.offline_model:
            return "offline"
        return "online"
