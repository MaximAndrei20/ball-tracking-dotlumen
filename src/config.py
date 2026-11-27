# Shared settings for the camera, football physics, video timeline, execution profiles, and file paths.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Nominal Intel RealSense D435i RGB pinhole calibration at 1280x720."""

    width: int = 1280
    height: int = 720
    hfov_deg: float = 69.4
    vfov_deg: float = 42.5

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

    @property
    def fy(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg / 2.0))

    @property
    def f_mean(self) -> float:
        return 0.5 * (self.fx + self.fy)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        # D435i factory metadata specifies zero Inverse Brown-Conrady distortion.
        return np.zeros(5, dtype=np.float64)


@dataclass(frozen=True)
class BallConfig:
    """Physical properties of a regulation FIFA Size 5 football."""

    diameter_m: float = 0.22
    radius_m: float = 0.11
    mass_kg: float = 0.43
    gravity_mps2: float = 9.81
    air_density: float = 1.225
    drag_coeff: float = 0.20
    rolling_friction: float = 0.03


@dataclass(frozen=True)
class VideoConfig:
    """Video properties, audited event windows, and empirical ball color bounds."""

    fps: float = 30.0
    num_frames: int = 462
    stationary_frames: tuple[int, int] = (0, 158)
    kick1_frames: tuple[int, int] = (161, 206)
    gap_frames: tuple[int, int] = (207, 378)
    kick2_frames: tuple[int, int] = (379, 461)
    ball_core_lab_a_min: int = 158
    ball_core_lab_l_min: int = 65
    ball_core_hsv_h_min: int = 152
    ball_core_hsv_s_min: int = 120
    ball_core_hsv_v_min: int = 115
    ball_refine_lab_a_min: int = 138
    ball_refine_hsv_s_min: int = 75
    ball_refine_hsv_h_low_max: int = 10
    ball_refine_hsv_h_high_min: int = 145


@dataclass(frozen=True)
class ProfileConfig:
    """Backend selection for local execution or optional cloud GPU upgrades."""

    name: str
    detector: str
    vo: str
    yolo_weights: str
    device: str


PROFILES: dict[str, ProfileConfig] = {
    "local": ProfileConfig(
        name="local",
        detector="yolo",
        vo="pycolmap",
        yolo_weights="yolo11s-seg.pt",
        device="cpu",
    ),
    "gpu": ProfileConfig(
        name="gpu",
        detector="sam3",
        vo="da3",
        yolo_weights="yolo11m-seg.pt",
        device="cuda",
    ),
}


@dataclass(frozen=True)
class PathsConfig:
    """Project directory layout and profile-scoped cache locations."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def video(self) -> Path:
        return self.root / "rgb.avi"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def plots_dir(self) -> Path:
        return self.outputs_dir / "plots"

    def cache_dir(self, profile: str = "local") -> Path:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; expected one of {tuple(PROFILES)}")
        return self.root / "cache" / profile

    def ensure_dirs(self, profile: str = "local") -> None:
        for prof in set(PROFILES) | {profile}:
            self.cache_dir(prof).mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Unified configuration container for all pipeline stages."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def get_profile(self, name: str = "local") -> ProfileConfig:
        if name not in PROFILES:
            raise ValueError(f"Unknown profile {name!r}; expected one of {tuple(PROFILES)}")
        return PROFILES[name]


CONFIG = PipelineConfig()
