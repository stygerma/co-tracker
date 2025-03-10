from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt
from sips.data import CameraPose


@dataclass
class Config:
    data_dir: str = "co-tracker/data"

    image_shape: tuple[int, int] = (616, 1116)

    angular_fov: int = 130
    range_resolution: float = 0.009571

    file_name: str = "sonarImageStructure"

    num_frames: int = 100
    synthetic_data: bool = False
    offline_model: bool = False
    force_redo: bool = True

    default_device: str = "cuda"

    # Synthetic keypoint parameters
    subpixel_accuracy: bool = False
    add_sonar_noise: bool = False
    true_keypoint_distances: list[float] = field(
        default_factory=lambda: [320, 300, 500, 330, 540, 560]
    )
    true_keypoint_angles: list[float] = field(
        default_factory=lambda: [-35, -35, -45, -35, -45, -45]
    )
    angular_step_size: int = 5
    keypoint_sigma: int = 2
    keypoint_intensity: int = 255

    check_visibilities: bool = True
    needed_visibility_ratio: float = (
        0.0  # 0.0 keeps all frames and only applies individual exclusion
    )

    # Manually selected keypoints for real data
    manual_keypoints: dict[str, npt.NDArray[np.float64]] = field(
        default_factory=lambda: {
            "sonarImageStructure": np.array(
                [
                    [434.2, 338.1],
                    [439.9, 164.3],
                    [507.9, 181.5],
                    [669.4, 378.1],
                    [582.9, 191.5],
                    [610.8, 175.1],
                ]
            ),
            "rerecordLimmatCartesian": np.array(
                [
                    [629.1, 182.4],
                    [477.2, 215.2],
                    [481.8, 569.7],
                    [541.1, 226.8],
                    [732.7, 327.4],
                    [573.0, 210.5],
                    [474.0, 192.6],
                ]
            ),
            "rerecordLimmatCartesianPoleApproach": np.array(
                [
                    [135.1, 359.3],
                    [678.2, 346.1],
                    [689.9, 332.8],
                    [713.3, 344.5],
                    [781.1, 333.6],
                    [619.8, 168.4],
                ],
                dtype=np.float64,
            ),
        }
    )

    init_pose: dict[str, CameraPose] = field(
        default_factory=lambda: {
            "default": CameraPose.neutral_pose(),
            "interpolated_sonarImageStructure": CameraPose(
                [-4.904834105602733, 2.577402348695038, -3.4409040879928843],
                [
                    -0.018202238128860054,
                    0.0037275872930560695,
                    0.5298373970162384,
                    0.8478956208875751,
                ],
            ),
            "sonarImageStructure": CameraPose(
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
            self.file_name = "fake_sonar"
            if self.subpixel_accuracy:
                self.file_name += "_subpixel"
            if self.add_sonar_noise:
                self.file_name += "_noisy"

    @property
    def num_tracks(self) -> int:
        if self.synthetic_data:
            return len(self.true_keypoint_distances)
        return len(self.manual_keypoints[self.file_name])

    @property
    def model_name(self) -> str:
        if self.offline_model:
            return "offline"
        return "online"
