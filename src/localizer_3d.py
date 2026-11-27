# Stage 2 3D ball localization and Stage 4 ground fusion. Computes camera-frame sphere depth, uncertainty, and floor intersections.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BallConfig, CameraConfig, CONFIG
from src.detector_2d import Stage1Result
from src.ego_motion import Stage3Result


@dataclass(frozen=True)
class SpherePose3D:
    """Single-frame 3D sphere position and uncertainty in OpenCV camera coordinates."""

    valid: bool
    method: str
    X_c: float
    Y_c: float
    Z_c: float
    distance_m: float
    sigma_X: float
    sigma_Y: float
    sigma_Z: float
    eccentricity_shift_px: float


def _invalid_pose() -> SpherePose3D:
    nan = float("nan")
    return SpherePose3D(False, "none", nan, nan, nan, nan, nan, nan, nan, nan)


def solve_penne_minor_axis(area_iso_pi: float, delta_m: float, f: float) -> float:
    """Solve Penne et al. (2019) area cubic w^3 + (dM^2 + f^2)w^2 - At^2 w - f^2 At^2 = 0 for b = sqrt(w)."""
    if area_iso_pi <= 0.0 or f <= 0.0:
        return float("nan")
    coeffs = [1.0, delta_m**2 + f**2, -(area_iso_pi**2), -(f**2) * (area_iso_pi**2)]
    roots = np.roots(coeffs)
    pos_real = [r for r in roots if r.real > 0.0]
    if not pos_real:
        return float("nan")
    best = min(pos_real, key=lambda r: abs(r.imag))
    return float(np.sqrt(best.real))


def localize_sphere_3d(
    u: float,
    v: float,
    a_px: float,
    b_px: float,
    area_px: float,
    camera: CameraConfig = CONFIG.camera,
    ball: BallConfig = CONFIG.ball,
    is_blurred: bool = False,
    occluded_person: bool = False,
    border_clipped: bool = False,
    bbox_xyxy: np.ndarray | None = None,
    ellipse_angle_deg: float = 0.0,
    sigma_d_px: float = 1.2,
    sigma_uv_px: float = 1.0,
) -> SpherePose3D:
    """Recover exact 3D camera-frame sphere coordinates and heteroscedastic uncertainty."""
    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(a_px) and np.isfinite(b_px)) or b_px <= 0.0:
        return _invalid_pose()

    f = camera.f_mean
    du = (u - camera.cx) * (f / camera.fx)
    dv = (v - camera.cy) * (f / camera.fy)
    delta_m = float(np.hypot(du, dv))
    sec_phi = float(np.sqrt(1.0 + (delta_m / f) ** 2))

    # Convert pixel ellipse axes along major/minor directions into isotropic focal units.
    ang_rad = np.radians(ellipse_angle_deg)
    maj_scale = f * float(np.hypot(np.cos(ang_rad) / camera.fx, np.sin(ang_rad) / camera.fy))
    min_scale = f * float(np.hypot(np.sin(ang_rad) / camera.fx, np.cos(ang_rad) / camera.fy))
    a_iso = a_px * maj_scale
    b_iso = b_px * min_scale
    aspect = a_iso / max(b_iso, 1e-6)

    if border_clipped:
        use_major = False
        if bbox_xyxy is not None and len(bbox_xyxy) == 4:
            _, y1, _, y2 = bbox_xyxy
            vert_clipped = (y1 <= 10.0) or (y2 >= camera.height - 11.0)
            use_major = vert_clipped and (aspect > 1.15)
        b = a_iso if use_major else b_iso
        method = "clipped_deweighted"
    elif (not is_blurred) and np.isfinite(area_px) and area_px > 0.0 and aspect <= 1.05 * sec_phi:
        area_iso_pi = (area_px / np.pi) * (f**2 / (camera.fx * camera.fy))
        b_cubic = solve_penne_minor_axis(area_iso_pi, delta_m, f)
        if np.isfinite(b_cubic) and b_cubic > 0.0:
            b = b_cubic
            method = "penne_cubic"
        else:
            b = b_iso
            method = "minor_axis"
    else:
        b = b_iso
        method = "minor_axis"

    R = ball.radius_m
    D = ball.diameter_m
    Z_c = float(R * np.sqrt(1.0 + (f / b) ** 2))
    scale_xy = (f * Z_c) / (f**2 + b**2)
    X_c = float(scale_xy * du)
    Y_c = float(scale_xy * dv)
    distance_m = float(np.sqrt(X_c**2 + Y_c**2 + Z_c**2))

    delta_q = (f**2 / (f**2 + b**2)) * delta_m
    ecc_shift = float(abs(delta_m - delta_q))

    sigma_Z = float((Z_c**2 / (f * D)) * sigma_d_px)
    sigma_XY = float((Z_c / f) * sigma_uv_px)
    sigma_X = sigma_XY
    sigma_Y = sigma_XY

    if border_clipped:
        sigma_Z *= 10.0
        sigma_X *= 3.0
        sigma_Y *= 3.0
    elif occluded_person:
        sigma_Z *= 10.0
        method = "occluded_deweighted"

    if delta_m > 0.7 * (camera.width / 2.0):
        sigma_Z *= 1.5

    return SpherePose3D(True, method, X_c, Y_c, Z_c, distance_m, sigma_X, sigma_Y, sigma_Z, ecc_shift)


def localize_stage1(
    res: Stage1Result,
    camera: CameraConfig = CONFIG.camera,
    ball: BallConfig = CONFIG.ball,
) -> pd.DataFrame:
    """Compute exact 3D camera-frame positions and uncertainties for all Stage 1 detections."""
    rows = []
    for i in range(len(res.frame_idx)):
        if res.detected[i]:
            pose = localize_sphere_3d(
                u=float(res.center_uv[i, 0]),
                v=float(res.center_uv[i, 1]),
                a_px=float(res.semi_axes_ab[i, 0]),
                b_px=float(res.semi_axes_ab[i, 1]),
                area_px=float(res.mask_area_px[i]),
                camera=camera,
                ball=ball,
                is_blurred=bool(res.is_blurred[i]),
                occluded_person=bool(res.occluded_person[i]),
                border_clipped=bool(res.border_clipped[i]),
                bbox_xyxy=res.bbox_xyxy[i],
                ellipse_angle_deg=float(res.ellipse_angle_deg[i]),
            )
        else:
            pose = _invalid_pose()

        rows.append(
            {
                "frame_idx": int(res.frame_idx[i]),
                "timestamp_s": float(res.timestamp_s[i]),
                "valid": pose.valid,
                "method": pose.method,
                "X_c": pose.X_c,
                "Y_c": pose.Y_c,
                "Z_c": pose.Z_c,
                "distance_m": pose.distance_m,
                "sigma_X": pose.sigma_X,
                "sigma_Y": pose.sigma_Y,
                "sigma_Z": pose.sigma_Z,
                "eccentricity_shift_px": pose.eccentricity_shift_px,
            }
        )
    return pd.DataFrame(rows)


def save_stage2_diagnostics(df: pd.DataFrame, plots_dir: Path) -> None:
    """Plot camera-frame 3D coordinates, distance with +/-2 sigma_Z bands, and sigma_Z growth."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames = df["frame_idx"].to_numpy()
    valid = df["valid"].to_numpy()
    dist = np.where(valid, df["distance_m"].to_numpy(), np.nan)
    z_c = np.where(valid, df["Z_c"].to_numpy(), np.nan)
    x_c = np.where(valid, df["X_c"].to_numpy(), np.nan)
    y_c = np.where(valid, df["Y_c"].to_numpy(), np.nan)
    sig_z = np.where(valid, df["sigma_Z"].to_numpy(), np.nan)
    methods = df["method"].to_numpy()

    g0, g1 = CONFIG.video.gap_frames
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.8), sharex=True)
    for ax in axes:
        ax.axvspan(g0, g1, color="gray", alpha=0.12)

    band = np.minimum(2.0 * sig_z, 1.5)
    axes[0].plot(frames, dist, label="Euclidean distance (m)", lw=1.6)
    axes[0].plot(frames, z_c, "--", label="Z_c forward depth (m)", lw=1.2, alpha=0.85)
    axes[0].fill_between(frames, dist - band, dist + band, alpha=0.2, label="+/-2 sigma_Z (capped 1.5m)")
    occ = valid & (methods == "occluded_deweighted")
    clip = valid & (methods == "clipped_deweighted")
    if np.any(occ):
        axes[0].scatter(frames[occ], dist[occ], color="orange", s=22, zorder=5, label="occluded_deweighted")
    if np.any(clip):
        axes[0].scatter(frames[clip], dist[clip], color="red", s=22, zorder=5, label="clipped_deweighted")
    axes[0].set_ylabel("Range (m)")
    axes[0].set_title("Stage 2: Camera-Frame 3D Sphere Localization & Uncertainty")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(frames, x_c, label="X_c right (m)", lw=1.5)
    axes[1].plot(frames, y_c, label="Y_c down (m)", lw=1.5)
    axes[1].axhline(0.0, color="gray", ls=":", lw=0.9)
    axes[1].set_ylabel("Lateral / Vertical (m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(frames, sig_z, color="tab:red", lw=1.4, label="sigma_Z (m)")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_ylabel("sigma_Z (m, log)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(plots_dir / "stage2_3d_raw.png", dpi=150)
    plt.close(fig)


def intersect_ray_with_plane_z(
    C_g: np.ndarray,
    d_g: np.ndarray,
    z_target: float,
) -> tuple[bool, np.ndarray, float]:
    """Intersect ground-frame unit ray C_g + lam * d_g with horizontal plane Z_g = z_target."""
    d_unit = d_g / max(float(np.linalg.norm(d_g)), 1e-12)
    if d_unit[2] >= -0.02:
        return False, np.full(3, np.nan, dtype=np.float64), float("nan")
    lam = float((z_target - C_g[2]) / d_unit[2])
    if not (0.3 < lam < 25.0):
        return False, np.full(3, np.nan, dtype=np.float64), float("nan")
    return True, C_g + lam * d_unit, lam


def fuse_sphere_with_ground_plane(
    s1: Stage1Result,
    df2: pd.DataFrame,
    s3: Stage3Result,
    camera: CameraConfig = CONFIG.camera,
    ball: BallConfig = CONFIG.ball,
    sigma_uv_px: float = 1.0,
    sigma_floor_m: float = 0.015,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fuse camera-frame sphere localization with terrace ray-ground intersection."""
    n = len(df2)
    R_ball = ball.radius_m
    f = camera.f_mean
    sigma_theta = sigma_uv_px / f

    P_g_sph = np.full((n, 3), np.nan, dtype=np.float64)
    P_g_gnd = np.full((n, 3), np.nan, dtype=np.float64)
    P_g_fused = np.full((n, 3), np.nan, dtype=np.float64)
    R_meas_g = np.full((n, 3, 3), np.nan, dtype=np.float64)
    cov_sph_all = np.full((n, 3, 3), np.nan, dtype=np.float64)
    cov_gnd_all = np.full((n, 3, 3), np.nan, dtype=np.float64)
    raw_lift = np.zeros(n, dtype=bool)
    gnd_ok = np.zeros(n, dtype=bool)

    for k in range(n):
        if not (s1.detected[k] and bool(df2.loc[k, "valid"]) and bool(s3.pose_valid[k])):
            continue
        Pc = df2.loc[k, ["X_c", "Y_c", "Z_c"]].to_numpy(dtype=np.float64)
        sigs = df2.loc[k, ["sigma_X", "sigma_Y", "sigma_Z"]].to_numpy(dtype=np.float64)
        R_gc = s3.R_gc[k]
        C_g = s3.C_g[k]

        d_c = Pc / max(float(np.linalg.norm(Pc)), 1e-12)
        d_g = R_gc @ d_c
        p_sph = R_gc @ Pc + C_g
        sig_rad = float(sigs[2]) / max(abs(float(d_c[2])), 0.2)
        sig_perp = float(0.5 * (sigs[0] + sigs[1]))
        cov_sph = (sig_rad**2) * np.outer(d_g, d_g) + (sig_perp**2) * (np.eye(3) - np.outer(d_g, d_g))
        P_g_sph[k] = p_sph
        cov_sph_all[k] = cov_sph

        hit, p_gnd, lam = intersect_ray_with_plane_z(C_g, d_g, R_ball)
        if hit:
            gnd_ok[k] = True
            P_g_gnd[k] = p_gnd

            h_xy = np.array([d_g[0], d_g[1], 0.0], dtype=np.float64)
            h_norm = float(np.linalg.norm(h_xy))
            h_hat = h_xy / h_norm if h_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            t_hat = np.array([-h_hat[1], h_hat[0], 0.0], dtype=np.float64)
            e_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            B = np.column_stack([h_hat, t_hat, e_z])

            dz_abs = max(abs(float(d_g[2])), 0.05)
            sig_lat = lam * sigma_theta
            sig_long = float(np.hypot((lam * sigma_theta) / dz_abs, sigma_floor_m / dz_abs))
            sig_z = 0.025

            if bool(s1.border_clipped[k]):
                sig_long *= 10.0
                sig_lat *= 10.0
                sig_z *= 5.0
            elif bool(s1.occluded_person[k]):
                sig_long *= 2.0
                sig_lat *= 2.0

            cov_gnd_all[k] = B @ np.diag([sig_long**2, sig_lat**2, sig_z**2]) @ B.T

        z_sigma = max(float(np.sqrt(cov_sph[2, 2])), 0.04)
        z_lift = (p_sph[2] - R_ball) / z_sigma
        if (not bool(s1.border_clipped[k])) and (not bool(s1.occluded_person[k])) and z_lift > 2.5:
            raw_lift[k] = True

    is_airborne = np.zeros(n, dtype=bool)
    for k in range(n):
        if raw_lift[k] and ((k > 0 and raw_lift[k - 1]) or (k + 1 < n and raw_lift[k + 1])):
            is_airborne[k] = True

    for k in range(n):
        if np.isnan(P_g_sph[k, 0]):
            continue
        if (not is_airborne[k]) and gnd_ok[k]:
            # Fuse strictly in the 2D floor plane so noisy sphere Z_g cannot leak along tilted rays into X_g.
            inv_sph_xy = np.linalg.inv(cov_sph_all[k, :2, :2])
            inv_gnd_xy = np.linalg.inv(cov_gnd_all[k, :2, :2])
            cov_xy = np.linalg.inv(inv_sph_xy + inv_gnd_xy)
            xy_f = cov_xy @ (inv_sph_xy @ P_g_sph[k, :2] + inv_gnd_xy @ P_g_gnd[k, :2])
            P_g_fused[k] = np.array([xy_f[0], xy_f[1], R_ball], dtype=np.float64)
            cov_f = np.zeros((3, 3), dtype=np.float64)
            cov_f[:2, :2] = cov_xy
            cov_f[2, 2] = cov_gnd_all[k, 2, 2]
            R_meas_g[k] = cov_f
        else:
            P_g_fused[k] = P_g_sph[k]
            R_meas_g[k] = cov_sph_all[k]

    df_out = df2.copy()
    df_out["X_g_sph"] = P_g_sph[:, 0]
    df_out["Y_g_sph"] = P_g_sph[:, 1]
    df_out["Z_g_sph"] = P_g_sph[:, 2]
    df_out["X_g_gnd"] = P_g_gnd[:, 0]
    df_out["Y_g_gnd"] = P_g_gnd[:, 1]
    df_out["X_g_fused"] = P_g_fused[:, 0]
    df_out["Y_g_fused"] = P_g_fused[:, 1]
    df_out["Z_g_fused"] = P_g_fused[:, 2]
    df_out["sig_x_g"] = np.sqrt(np.maximum(R_meas_g[:, 0, 0], 0.0))
    df_out["sig_y_g"] = np.sqrt(np.maximum(R_meas_g[:, 1, 1], 0.0))
    df_out["sig_z_g"] = np.sqrt(np.maximum(R_meas_g[:, 2, 2], 0.0))
    df_out["is_airborne"] = is_airborne
    return df_out, R_meas_g


def run_stage2(profile_name: str = "local") -> pd.DataFrame:
    """Execute Stage 2 3D sphere localization from Stage 1 cache and save CSV + diagnostics."""
    CONFIG.paths.ensure_dirs(profile_name)
    cache_dir = CONFIG.paths.cache_dir(profile_name)
    s1 = Stage1Result.load(cache_dir)
    df = localize_stage1(s1)
    df.to_csv(cache_dir / "stage2_3d.csv", index=False)
    save_stage2_diagnostics(df, CONFIG.paths.plots_dir)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Exact 3D camera-frame sphere localization.")
    parser.add_argument("--profile", default="local", choices=("local", "gpu"))
    args = parser.parse_args()

    df = run_stage2(args.profile)
    valid = df["valid"].to_numpy()
    s0, s1 = CONFIG.video.stationary_frames
    stat_mask = valid & (df["frame_idx"].to_numpy() >= s0) & (df["frame_idx"].to_numpy() <= s1)
    print(f"Valid 3D poses: {int(valid.sum())}/{len(df)}")
    print("Methods:", df.loc[valid, "method"].value_counts().to_dict())
    print(
        f"Pre-kick distance ({s0}-{s1}): median={df.loc[stat_mask, 'distance_m'].median():.3f} m, "
        f"std={df.loc[stat_mask, 'distance_m'].std():.3f} m"
    )
    print(
        f"Overall distance range: {df.loc[valid, 'distance_m'].min():.3f} m .. {df.loc[valid, 'distance_m'].max():.3f} m | "
        f"Median sigma_Z: {df.loc[valid, 'sigma_Z'].median():.3f} m"
    )


if __name__ == "__main__":
    main()
