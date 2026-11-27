# Stage 3 camera ego-motion and ground alignment. Runs pycolmap SfM, SIFT fallback, DA3/MegaSaM loading, scale anchoring, and floor plane fitting.

from __future__ import annotations

import argparse
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from src.config import CONFIG, ProfileConfig
from src.detector_2d import Stage1Result


@dataclass
class Stage3Result:
    """Metric camera poses, ground-plane alignment, and static terrace points."""

    R_wc: np.ndarray
    C_w: np.ndarray
    R_gw: np.ndarray
    t_gw: np.ndarray
    R_gc: np.ndarray
    C_g: np.ndarray
    scale_alpha: float
    ground_plane_abcd: np.ndarray
    sparse_points_g: np.ndarray
    stat_ball_tri_g: np.ndarray
    pose_valid: np.ndarray
    vo_mode: np.ndarray

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / "stage3_ego.npz",
            R_wc=self.R_wc,
            C_w=self.C_w,
            R_gw=self.R_gw,
            t_gw=self.t_gw,
            R_gc=self.R_gc,
            C_g=self.C_g,
            scale_alpha=np.float64(self.scale_alpha),
            ground_plane_abcd=self.ground_plane_abcd,
            sparse_points_g=self.sparse_points_g,
            stat_ball_tri_g=self.stat_ball_tri_g,
            pose_valid=self.pose_valid,
            vo_mode=self.vo_mode,
        )

    @classmethod
    def load(cls, cache_dir: Path) -> Stage3Result:
        with np.load(cache_dir / "stage3_ego.npz", allow_pickle=False) as d:
            return cls(
                R_wc=d["R_wc"],
                C_w=d["C_w"],
                R_gw=d["R_gw"],
                t_gw=d["t_gw"],
                R_gc=d["R_gc"],
                C_g=d["C_g"],
                scale_alpha=float(d["scale_alpha"]),
                ground_plane_abcd=d["ground_plane_abcd"],
                sparse_points_g=d["sparse_points_g"],
                stat_ball_tri_g=d["stat_ball_tri_g"],
                pose_valid=d["pose_valid"],
                vo_mode=d["vo_mode"],
            )


def chain_relative_pose(
    R_wc_prev: np.ndarray,
    C_prev: np.ndarray,
    R_rel: np.ndarray,
    t_rel: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Chain camera-to-world pose from relative transform P_curr = R_rel @ P_prev + s * t_rel."""
    R_wc = R_wc_prev @ R_rel.T
    C_curr = C_prev - scale * (R_wc @ t_rel)
    return R_wc, C_curr


def triangulate_stationary_ball(
    R_cw_list: list[np.ndarray] | np.ndarray,
    t_cw_list: list[np.ndarray] | np.ndarray,
    uv_list: list[np.ndarray] | np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-view DLT SVD triangulation of a static 3D point across calibrated cameras."""
    rows = []
    for R_cw, t_cw, (u, v) in zip(R_cw_list, t_cw_list, uv_list):
        P = K @ np.hstack([R_cw, np.asarray(t_cw).reshape(3, 1)])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])

    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    X_h = vt[-1]
    X_tri = X_h[:3] / X_h[3]

    errs = []
    for R_cw, t_cw, (u, v) in zip(R_cw_list, t_cw_list, uv_list):
        Xc = R_cw @ X_tri + np.asarray(t_cw).reshape(3)
        proj = K @ Xc
        errs.append(float(np.hypot(proj[0] / proj[2] - u, proj[1] / proj[2] - v)))
    return X_tri, np.asarray(errs, dtype=np.float64)


def solve_scale_soft_l1(
    X_tri: np.ndarray,
    C_unscaled: np.ndarray,
    dist_m: np.ndarray,
    sigma_z: np.ndarray | None = None,
) -> float:
    """Solve unbiased global scale alpha aligning triangulated distances to metric sphere ranges."""
    unscaled_d = np.linalg.norm(X_tri[None, :] - C_unscaled, axis=1)
    valid = unscaled_d > 1e-6
    ratios = dist_m[valid] / unscaled_d[valid]
    alpha0 = float(np.median(ratios))
    if np.sum(valid) < 2:
        return alpha0

    w = np.ones(int(np.sum(valid)), dtype=np.float64) if sigma_z is None else 1.0 / np.maximum(sigma_z[valid], 0.05)
    d_obs = dist_m[valid]
    u_obs = unscaled_d[valid]

    res = least_squares(
        lambda x: w * (x[0] * u_obs - d_obs),
        x0=[alpha0],
        bounds=([1e-5], [1e5]),
        loss="soft_l1",
        f_scale=0.25,
    )
    return float(res.x[0])


def detect_stationary_ball_window(
    R_wc_unscaled: np.ndarray,
    C_w_unscaled: np.ndarray,
    pose_valid: np.ndarray,
    s1: Stage1Result,
    df2: pd.DataFrame,
    K: np.ndarray,
) -> tuple[list[int], np.ndarray | None, np.ndarray]:
    """Automatically find the strongest contiguous stationary-ball span via sliding-window DLT."""
    n = len(pose_valid)
    valid_obs = [
        k for k in range(n)
        if pose_valid[k] and s1.detected[k] and bool(df2.loc[k, "valid"]) and not bool(s1.border_clipped[k])
    ]
    if len(valid_obs) < 15 or not np.any(pose_valid):
        return [], None, np.array([], dtype=np.float64)

    C_valid = C_w_unscaled[pose_valid]
    cam_span_u = float(np.linalg.norm(C_valid.max(axis=0) - C_valid.min(axis=0)))
    if cam_span_u < 1e-6:
        return [], None, np.array([], dtype=np.float64)

    runs: list[list[int]] = [[valid_obs[0]]]
    for f in valid_obs[1:]:
        if f - runs[-1][-1] <= 6:
            runs[-1].append(f)
        else:
            runs.append([f])

    best_win: list[int] = []
    best_score = -1.0
    for run in runs:
        if len(run) < 15:
            continue
        for start in range(0, len(run) - 14, 8):
            sub = run[start : min(start + 30, len(run))]
            sub_span = float(np.linalg.norm(C_w_unscaled[sub].max(axis=0) - C_w_unscaled[sub].min(axis=0)))
            if sub_span < max(0.05 * cam_span_u, 1e-5):
                continue
            R_cw = [R_wc_unscaled[k].T for k in sub]
            t_cw = [-R_wc_unscaled[k].T @ C_w_unscaled[k] for k in sub]
            uv = [s1.center_uv[k] for k in sub]
            X_t, errs = triangulate_stationary_ball(R_cw, t_cw, uv, K)
            if float(np.median(errs)) >= 3.0 or float(np.percentile(errs, 90)) >= 6.0:
                continue

            inliers = []
            for k in run:
                Xc = R_wc_unscaled[k].T @ (X_t - C_w_unscaled[k])
                if Xc[2] <= 0.05:
                    continue
                pr = K @ Xc
                err_k = float(np.hypot(pr[0] / pr[2] - s1.center_uv[k, 0], pr[1] / pr[2] - s1.center_uv[k, 1]))
                if err_k < 5.0:
                    inliers.append(k)
            if len(inliers) < 8:
                continue
            inl_span = float(np.linalg.norm(C_w_unscaled[inliers].max(axis=0) - C_w_unscaled[inliers].min(axis=0)))
            if inl_span < max(0.05 * cam_span_u, 1e-5):
                continue
            score = len(inliers) * inl_span
            if score > best_score:
                best_score = score
                best_win = inliers

    if not best_win:
        return [], None, np.array([], dtype=np.float64)

    X_tri, reproj_errs = triangulate_stationary_ball(
        [R_wc_unscaled[k].T for k in best_win],
        [-R_wc_unscaled[k].T @ C_w_unscaled[k] for k in best_win],
        [s1.center_uv[k] for k in best_win],
        K,
    )
    return best_win, X_tri, reproj_errs


def solve_scale_floor_contact(
    R_wc_unscaled: np.ndarray,
    C_w_unscaled: np.ndarray,
    points_w_unscaled: np.ndarray,
    pose_valid: np.ndarray,
    s1: Stage1Result,
    df2: pd.DataFrame,
    ball_radius_m: float = CONFIG.ball.radius_m,
) -> tuple[float, int]:
    """Recover metric scale from moving ground-rolling ball detections intersecting the unscaled floor."""
    valid_cam = np.where(pose_valid)[0]
    if len(valid_cam) == 0 or len(points_w_unscaled) < 10:
        raise RuntimeError("Insufficient poses or 3D points for floor-contact metric scale fallback.")
    f0 = int(valid_cam[0])
    n_u, d_u, _ = fit_ground_plane_ransac(points_w_unscaled, C_w_unscaled[f0], R_wc_unscaled[f0], unscaled=True)

    unclip_idx = [
        k for k in range(len(pose_valid))
        if pose_valid[k] and s1.detected[k] and bool(df2.loc[k, "valid"]) and not bool(s1.border_clipped[k])
    ]
    clean_idx = [k for k in unclip_idx if df2.loc[k, "method"] not in ("occluded_deweighted", "clipped_deweighted")]
    use_idx = clean_idx if len(clean_idx) >= 5 else unclip_idx

    hu_list, nd_list, d_sph_list, sig_list = [], [], [], []
    for k in use_idx:
        Pc = df2.loc[k, ["X_c", "Y_c", "Z_c"]].to_numpy(dtype=np.float64)
        d_w = R_wc_unscaled[k] @ (Pc / max(float(np.linalg.norm(Pc)), 1e-12))
        nd = float(np.dot(n_u, d_w))
        hu = float(np.dot(n_u, C_w_unscaled[k]) + d_u)
        if nd < -0.05 and hu > 1e-4:
            hu_list.append(hu)
            nd_list.append(nd)
            d_sph_list.append(float(df2.loc[k, "distance_m"]))
            sig_list.append(float(df2.loc[k, "sigma_Z"]))

    if len(hu_list) < 3:
        raise RuntimeError(f"Insufficient valid floor rays ({len(hu_list)}) for metric scale fallback.")

    hu_arr = np.asarray(hu_list, dtype=np.float64)
    nd_arr = np.asarray(nd_list, dtype=np.float64)
    d_arr = np.asarray(d_sph_list, dtype=np.float64)
    w = 1.0 / np.maximum(np.asarray(sig_list, dtype=np.float64), 0.05)

    alpha0 = max(1e-5, float(np.median((d_arr * (-nd_arr) + ball_radius_m) / hu_arr)))
    res = least_squares(
        lambda x: w * (((ball_radius_m - x[0] * hu_arr) / nd_arr) - d_arr),
        x0=[alpha0],
        bounds=([1e-5], [1e5]),
        loss="soft_l1",
        f_scale=0.25,
    )
    return float(res.x[0]), len(hu_list)


def anchor_metric_scale(
    R_wc_unscaled: np.ndarray,
    C_w_unscaled: np.ndarray,
    pose_valid: np.ndarray,
    s1: Stage1Result,
    df2: pd.DataFrame,
    K: np.ndarray,
    stat_range: tuple[int, int] | None = None,
    points_w_unscaled: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    """Anchor monocular reconstruction scale automatically via stationary-ball DLT or floor contact."""
    if stat_range is not None:
        s0, s1_end = stat_range
        tri_idx = [
            k for k in range(s0, min(s1_end + 1, len(pose_valid)))
            if pose_valid[k] and s1.detected[k] and bool(df2.loc[k, "valid"])
        ]
        if len(tri_idx) < 4:
            raise RuntimeError(f"Insufficient valid stationary frames ({len(tri_idx)}) for scale anchoring.")
        R_cw_list = [R_wc_unscaled[k].T for k in tri_idx]
        t_cw_list = [-R_wc_unscaled[k].T @ C_w_unscaled[k] for k in tri_idx]
        uv_list = [s1.center_uv[k] for k in tri_idx]
        X_tri, reproj_errs = triangulate_stationary_ball(R_cw_list, t_cw_list, uv_list, K)
    else:
        tri_idx, X_tri, reproj_errs = detect_stationary_ball_window(
            R_wc_unscaled, C_w_unscaled, pose_valid, s1, df2, K
        )

    if len(tri_idx) >= 8 and X_tri is not None:
        clean_mask = np.array(
            [df2.loc[k, "method"] not in ("occluded_deweighted", "clipped_deweighted") for k in tri_idx],
            dtype=bool,
        )
        use_idx = np.array(tri_idx)[clean_mask] if np.sum(clean_mask) >= 5 else np.array(tri_idx)
        C_sub = C_w_unscaled[use_idx]
        dist_sub = df2.loc[use_idx, "distance_m"].to_numpy(dtype=np.float64)
        sig_sub = df2.loc[use_idx, "sigma_Z"].to_numpy(dtype=np.float64)
        alpha = solve_scale_soft_l1(X_tri, C_sub, dist_sub, sig_sub)
        med_err = float(np.median(reproj_errs))
        if stat_range is None:
            print(f"Scale anchor: Tier A auto-stationary F{tri_idx[0]}..F{tri_idx[-1]} ({len(tri_idx)}f, reproj={med_err:.2f}px)")
        return alpha, alpha * X_tri, med_err

    if points_w_unscaled is not None:
        alpha, n_rays = solve_scale_floor_contact(
            R_wc_unscaled, C_w_unscaled, points_w_unscaled, pose_valid, s1, df2, CONFIG.ball.radius_m
        )
        print(f"Scale anchor: Tier B floor-contact ({n_rays} moving ball rays)")
        return alpha, np.full(3, np.nan, dtype=np.float64), float("nan")

    raise RuntimeError("No stationary ball segment found and no 3D points supplied for floor-contact scale fallback.")


def fit_ground_plane_ransac(
    points_w: np.ndarray,
    C_w0: np.ndarray,
    R_wc0: np.ndarray,
    seed: int = 42,
    inlier_thresh_m: float = 0.04,
    unscaled: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit dominant horizontal terrace floor plane n^T P + d = 0 with upward normal."""
    R_cw0 = R_wc0.T
    t_cw0 = -R_cw0 @ C_w0
    pts_c0 = (R_cw0 @ points_w.T).T + t_cw0

    if unscaled:
        pos_z = pts_c0[pts_c0[:, 2] > 1e-3, 2]
        med_z = float(np.median(pos_z)) if len(pos_z) > 0 else 1.0
        cand_mask = (pts_c0[:, 2] > 0.15 * med_z) & (pts_c0[:, 2] < 2.5 * med_z)
        thresh = 0.015 * med_z
    else:
        # Restrict RANSAC sampling to terrace-range points below the upper wall horizon.
        cand_mask = (pts_c0[:, 2] > 1.0) & (pts_c0[:, 2] < 10.0) & (pts_c0[:, 1] > -1.3)
        thresh = inlier_thresh_m
        med_z = 1.0

    cand_pts = points_w[cand_mask] if np.sum(cand_mask) >= 30 else points_w
    cam_up = -R_wc0[:, 1]

    rng = np.random.default_rng(seed)
    best_inl = None
    n_cands = len(cand_pts)

    for _ in range(1500):
        p0, p1, p2 = cand_pts[rng.choice(n_cands, 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            continue
        n /= norm
        if float(np.dot(n, cam_up)) < 0.0:
            n = -n
        if float(np.dot(n, cam_up)) < 0.70:
            continue
        d = -float(np.dot(n, p0))
        h_cam = float(np.dot(n, C_w0)) + d
        if unscaled:
            if h_cam <= 0.05 * med_z:
                continue
        elif not (0.6 < h_cam < 2.6):
            continue
        inl = np.abs(cand_pts @ n + d) < thresh
        if best_inl is None or int(inl.sum()) > int(best_inl.sum()):
            best_inl = inl

    if best_inl is None or int(best_inl.sum()) < 10:
        raise RuntimeError("Failed to find a valid ground plane via RANSAC.")

    in_pts = cand_pts[best_inl]
    centroid = in_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(in_pts - centroid)
    n_up = vt[-1]
    if float(np.dot(n_up, cam_up)) < 0.0:
        n_up = -n_up
    d_plane = -float(np.dot(n_up, centroid))
    all_inl = np.abs(points_w @ n_up + d_plane) < thresh
    return n_up, d_plane, all_inl


def build_ground_alignment(
    n_up: np.ndarray,
    d_plane: float,
    R_wc0: np.ndarray,
    C_w0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct right-handed ground frame G with Z_g=0 on floor and X_g along camera 0 heading."""
    ez = n_up / np.linalg.norm(n_up)
    z_cam0 = R_wc0[:, 2]
    ex = z_cam0 - float(np.dot(z_cam0, ez)) * ez
    if float(np.linalg.norm(ex)) < 1e-6:
        ex = R_wc0[:, 0] - float(np.dot(R_wc0[:, 0], ez)) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)

    R_gw = np.vstack([ex, ey, ez])
    p_orig = C_w0 - (float(np.dot(ez, C_w0)) + d_plane) * ez
    t_gw = -R_gw @ p_orig
    return R_gw, t_gw


def _interpolate_poses(
    poses_map: dict[int, tuple[np.ndarray, np.ndarray]],
    n_frames: int,
    vo_mode: np.ndarray,
    pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill any non-localized frames using SE(3) SLERP between bracketing poses."""
    reg_idx = sorted(poses_map.keys())
    R_out = np.zeros((n_frames, 3, 3), dtype=np.float64)
    C_out = np.zeros((n_frames, 3), dtype=np.float64)

    for k in reg_idx:
        R_out[k], C_out[k] = poses_map[k]
        pose_valid[k] = True

    if len(reg_idx) < 2:
        for k in range(n_frames):
            R_out[k], C_out[k] = poses_map[reg_idx[0]]
            pose_valid[k] = True
            if k != reg_idx[0]:
                vo_mode[k] = "sfm_interp"
        return R_out, C_out

    rots = Rotation.from_matrix(np.array([poses_map[k][0] for k in reg_idx]))
    slerp = Slerp(reg_idx, rots)
    Cs = np.array([poses_map[k][1] for k in reg_idx])

    for f in range(n_frames):
        if pose_valid[f]:
            continue
        if f <= reg_idx[0]:
            R_out[f], C_out[f] = poses_map[reg_idx[0]]
        elif f >= reg_idx[-1]:
            R_out[f], C_out[f] = poses_map[reg_idx[-1]]
        else:
            R_out[f] = slerp(f).as_matrix()
            for dim in range(3):
                C_out[f, dim] = float(np.interp(f, reg_idx, Cs[:, dim]))
        pose_valid[f] = True
        vo_mode[f] = "sfm_interp"
    return R_out, C_out


def _filter_outlier_keyframe_poses(
    kf_poses: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[set[int], np.ndarray, float]:
    """Identify SfM keyframes whose camera centers jump implausibly far from the main trajectory cluster."""
    if len(kf_poses) < 6:
        return set(), np.zeros(3, dtype=np.float64), float("inf")
    keys = sorted(kf_poses.keys())
    centers = np.asarray([kf_poses[k][1] for k in keys], dtype=np.float64)
    med_c = np.median(centers, axis=0)
    d_med = np.linalg.norm(centers - med_c, axis=1)
    mad_scale = max(float(np.median(d_med)), 1e-3)
    p90_span = max(float(np.percentile(d_med, 90)), mad_scale)
    max_rad = max(6.0 * mad_scale, 3.0 * p90_span)
    bad = {keys[i] for i, dist in enumerate(d_med) if dist > max_rad}
    return bad, med_c, max_rad


class BaseEgoMotion(ABC):
    """Abstract interface for Stage 3 unscaled monocular camera pose & static map recovery."""

    @abstractmethod
    def estimate(
        self,
        video_path: Path,
        s1: Stage1Result,
        cache_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (R_wc_unscaled, C_w_unscaled, points_w_unscaled, pose_valid, vo_mode)."""


def select_keyframes_optical_flow(
    video_path: Path,
    s1: Stage1Result,
    min_step: int = 4,
    max_step: int = 18,
    flow_thresh_px: float = 12.0,
    min_track_ratio: float = 0.55,
    max_keyframes: int = 95,
) -> list[int]:
    """Select SfM keyframes adaptively from static-masked background optical flow."""
    n_frames = len(s1.frame_idx)
    if n_frames <= 2:
        return list(range(n_frames))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    scale_down = 0.5
    kfs: list[int] = [0]
    prev_gray: np.ndarray | None = None
    prev_pts: np.ndarray | None = None

    def _corners(gray_s: np.ndarray, fidx: int) -> np.ndarray | None:
        sm = Stage1Result.unpack_static_mask(s1.static_masks_packed[fidx]).astype(np.uint8) * 255
        sm_s = cv2.resize(sm, (gray_s.shape[1], gray_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.goodFeaturesToTrack(gray_s, maxCorners=200, qualityLevel=0.01, minDistance=12, mask=sm_s)

    for f in range(n_frames):
        ok, frm = cap.read()
        if not ok:
            break
        gray_s = cv2.resize(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale_down, fy=scale_down)
        if f == 0:
            prev_gray = gray_s
            prev_pts = _corners(gray_s, f)
            continue
        if f - kfs[-1] < min_step:
            continue

        trigger = (f - kfs[-1]) >= max_step
        if not trigger and prev_pts is not None and len(prev_pts) >= 15 and prev_gray is not None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_s, prev_pts, None)
            good = st.ravel() == 1
            if int(good.sum()) < min_track_ratio * len(prev_pts):
                trigger = True
            else:
                disp_px = np.linalg.norm(nxt[good].reshape(-1, 2) - prev_pts[good].reshape(-1, 2), axis=1) / scale_down
                if float(np.median(disp_px)) >= flow_thresh_px:
                    trigger = True

        if trigger:
            kfs.append(f)
            prev_gray = gray_s
            prev_pts = _corners(gray_s, f)
    cap.release()

    if kfs[-1] != n_frames - 1:
        kfs.append(n_frames - 1)
    if len(kfs) > max_keyframes:
        keep_idx = np.unique(np.round(np.linspace(0, len(kfs) - 1, max_keyframes)).astype(int))
        kfs = [kfs[i] for i in keep_idx]
    return kfs


class PyColmapEgoMotion(BaseEgoMotion):
    """Static-masked pycolmap bundle-adjusted SfM on adaptive keyframes + LK PnP propagation."""

    def _localize_all_frames(
        self,
        video_path: Path,
        s1: Stage1Result,
        rec: pycolmap.Reconstruction,
        n_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = CONFIG.camera.K
        dist = CONFIG.camera.dist_coeffs
        kf_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        kf_obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for im in rec.images.values():
            if not im.has_pose:
                continue
            fidx = int(im.name.split("_")[1].split(".")[0])
            cfw = im.cam_from_world()
            R_cw = cfw.rotation.matrix()
            t_cw = cfw.translation
            R_wc = R_cw.T
            C_w = -R_wc @ t_cw
            kf_poses[fidx] = (R_wc, C_w)

            pts2d, pts3d = [], []
            for p2d in im.points2D:
                if p2d.has_point3D():
                    pt3 = rec.points3D[p2d.point3D_id]
                    if pt3.error < 2.5:
                        pts2d.append(p2d.xy)
                        pts3d.append(pt3.xyz)
            if len(pts2d) >= 15:
                kf_obs[fidx] = (np.asarray(pts2d, dtype=np.float32), np.asarray(pts3d, dtype=np.float64))

        bad_kfs, med_c, max_rad = _filter_outlier_keyframe_poses(kf_poses)
        if bad_kfs:
            for k in bad_kfs:
                kf_poses.pop(k, None)
                kf_obs.pop(k, None)
            print(f"Filtered {len(bad_kfs)} degenerate SfM keyframe(s): {sorted(bad_kfs)}")

        reg_kfs = sorted(kf_poses.keys())
        poses_map: dict[int, tuple[np.ndarray, np.ndarray]] = dict(kf_poses)
        pose_valid = np.zeros(n_frames, dtype=bool)
        vo_mode = np.full(n_frames, "none", dtype="U16")
        for k in reg_kfs:
            pose_valid[k] = True
            vo_mode[k] = "sfm_kf"

        cap = cv2.VideoCapture(str(video_path))
        grays: list[np.ndarray] = []
        while len(grays) < n_frames:
            ret, frm = cap.read()
            if not ret:
                break
            grays.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
        cap.release()

        # Propagate 2D-3D tracks from nearest registered keyframe to bracketed non-keyframes.
        for i in range(len(reg_kfs)):
            k_a = reg_kfs[i]
            k_b = reg_kfs[i + 1] if i + 1 < len(reg_kfs) else n_frames - 1
            span_end = (k_a + k_b) // 2 if i + 1 < len(reg_kfs) else n_frames - 1
            targets_fwd = [f for f in range(k_a + 1, span_end + 1) if not pose_valid[f]]
            targets_bwd = (
                [f for f in range(k_b - 1, span_end, -1) if not pose_valid[f]]
                if i + 1 < len(reg_kfs)
                else []
            )

            for k_ref, seq in ((k_a, targets_fwd), (k_b if i + 1 < len(reg_kfs) else k_a, targets_bwd)):
                if not seq or k_ref not in kf_obs:
                    continue
                p2_curr, p3_curr = kf_obs[k_ref]
                g_prev = grays[k_ref]
                R_wc_prev, C_prev = kf_poses[k_ref]
                C_step_prev = C_prev.copy()
                rvec_guess, _ = cv2.Rodrigues(R_wc_prev.T)
                tvec_guess = (-R_wc_prev.T @ C_prev).reshape(3, 1)

                for f in seq:
                    if len(p2_curr) < 20:
                        break
                    g_curr = grays[f]
                    p_fwd, st1, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_curr, p2_curr, None)
                    p_bwd, st2, _ = cv2.calcOpticalFlowPyrLK(g_curr, g_prev, p_fwd, None)
                    fb_err = np.linalg.norm(p_bwd - p2_curr, axis=1)
                    sm = Stage1Result.unpack_static_mask(s1.static_masks_packed[f])
                    ix = np.clip(np.round(p_fwd[:, 0]).astype(int), 0, sm.shape[1] - 1)
                    iy = np.clip(np.round(p_fwd[:, 1]).astype(int), 0, sm.shape[0] - 1)
                    keep = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0) & sm[iy, ix]
                    if int(keep.sum()) < 20:
                        break

                    p2_curr = p_fwd[keep]
                    p3_curr = p3_curr[keep]
                    g_prev = g_curr

                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        p3_curr,
                        p2_curr.astype(np.float64),
                        K,
                        dist,
                        rvec=rvec_guess.copy(),
                        tvec=tvec_guess.copy(),
                        useExtrinsicGuess=True,
                        iterationsCount=200,
                        reprojectionError=2.5,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if ok and inl is not None and len(inl) >= 15:
                        idx_in = inl.ravel()
                        rvec, tvec = cv2.solvePnPRefineLM(
                            p3_curr[idx_in], p2_curr[idx_in].astype(np.float64), K, dist, rvec, tvec
                        )
                        R_cw, _ = cv2.Rodrigues(rvec)
                        R_wc = R_cw.T
                        C_w = (-R_wc @ tvec).ravel()
                        if np.linalg.norm(C_w - med_c) > max_rad or np.linalg.norm(C_w - C_step_prev) > 0.6 * max_rad:
                            break
                        poses_map[f] = (R_wc, C_w)
                        pose_valid[f] = True
                        vo_mode[f] = "sfm_pnp"
                        C_step_prev = C_w
                        rvec_guess, tvec_guess = rvec, tvec

        R_all, C_all = _interpolate_poses(poses_map, n_frames, vo_mode, pose_valid)
        return R_all, C_all, pose_valid, vo_mode

    def estimate(
        self,
        video_path: Path,
        s1: Stage1Result,
        cache_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_frames = len(s1.frame_idx)
        kfs = select_keyframes_optical_flow(video_path, s1)
        print(f"Selected {len(kfs)} adaptive optical-flow keyframes across {n_frames} frames")
        work = cache_dir / "_colmap_tmp"
        if work.exists():
            shutil.rmtree(work)
        img_dir, msk_dir, out_dir = work / "images", work / "masks", work / "sparse"
        img_dir.mkdir(parents=True)
        msk_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        db_path = work / "db.db"

        try:
            cap = cv2.VideoCapture(str(video_path))
            for idx in kfs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                name = f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                sm = Stage1Result.unpack_static_mask(s1.static_masks_packed[idx]).astype(np.uint8) * 255
                cv2.imwrite(str(msk_dir / f"{name}.png"), sm)
            cap.release()

            cam = CONFIG.camera
            reader_opts = pycolmap.ImageReaderOptions()
            reader_opts.camera_model = "PINHOLE"
            reader_opts.camera_params = f"{cam.fx},{cam.fy},{cam.cx},{cam.cy}"
            reader_opts.mask_path = str(msk_dir)

            ext_opts = pycolmap.FeatureExtractionOptions()
            ext_opts.use_gpu = False
            ext_opts.sift.max_num_features = 2048

            pycolmap.extract_features(
                db_path,
                img_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options=reader_opts,
                extraction_options=ext_opts,
            )

            match_opts = pycolmap.FeatureMatchingOptions()
            match_opts.use_gpu = False
            pycolmap.match_exhaustive(db_path, matching_options=match_opts)

            rec = None
            for tri_deg in (6.0, 4.0):
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                po = pycolmap.IncrementalPipelineOptions()
                po.ba_refine_focal_length = False
                po.ba_refine_principal_point = False
                po.ba_refine_extra_params = False
                po.multiple_models = True
                po.min_model_size = 10
                po.max_num_models = 5
                po.random_seed = 42
                po.mapper.init_min_tri_angle = tri_deg
                po.mapper.abs_pose_refine_focal_length = False
                po.mapper.abs_pose_refine_extra_params = False

                recons = pycolmap.incremental_mapping(db_path, img_dir, out_dir, options=po)
                if not recons:
                    continue
                cand = max(recons.values(), key=lambda r: r.num_reg_images())
                if rec is None or cand.num_reg_images() > rec.num_reg_images():
                    rec = cand
                if rec.num_reg_images() >= 0.9 * len(kfs):
                    break

            if rec is None:
                raise RuntimeError("pycolmap produced no 3D reconstruction.")
            if rec.num_reg_images() < 0.9 * len(kfs):
                raise RuntimeError(
                    f"pycolmap registered only {rec.num_reg_images()}/{len(kfs)} keyframes."
                )

            pts_w = np.asarray(
                [p.xyz for p in rec.points3D.values() if p.error < 2.0 and p.track.length() >= 4],
                dtype=np.float64,
            )
            R_all, C_all, pose_valid, vo_mode = self._localize_all_frames(video_path, s1, rec, n_frames)
            return R_all, C_all, pts_w, pose_valid, vo_mode
        finally:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)


class SiftVoEgoMotion(BaseEgoMotion):
    """Pure OpenCV fallback: static-masked SIFT + USAC_MAGSAC Essential / SO(3) Homography."""

    def estimate(
        self,
        video_path: Path,
        s1: Stage1Result,
        cache_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = CONFIG.camera.K
        K_inv = np.linalg.inv(K)
        n_frames = len(s1.frame_idx)
        sift = cv2.SIFT_create(nfeatures=1500)
        bf = cv2.BFMatcher(cv2.NORM_L2)

        cap = cv2.VideoCapture(str(video_path))
        R_all = np.zeros((n_frames, 3, 3), dtype=np.float64)
        C_all = np.zeros((n_frames, 3), dtype=np.float64)
        pose_valid = np.ones(n_frames, dtype=bool)
        vo_mode = np.full(n_frames, "vo_essential", dtype="U16")

        R_all[0] = np.eye(3)
        C_all[0] = np.zeros(3)
        pts_cloud: list[np.ndarray] = []

        ret, f0 = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Cannot read {video_path}")
        g_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        m_prev = Stage1Result.unpack_static_mask(s1.static_masks_packed[0]).astype(np.uint8) * 255
        kp_prev, des_prev = sift.detectAndCompute(g_prev, m_prev)
        prev_idx = 0

        for idx in range(1, n_frames):
            ret, frm = cap.read()
            if not ret:
                break
            g_curr = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            m_curr = Stage1Result.unpack_static_mask(s1.static_masks_packed[idx]).astype(np.uint8) * 255
            kp_curr, des_curr = sift.detectAndCompute(g_curr, m_curr)

            if des_prev is None or des_curr is None or len(des_prev) < 20 or len(des_curr) < 20:
                R_all[idx], C_all[idx] = R_all[idx - 1], C_all[idx - 1]
                vo_mode[idx] = "vo_hold"
                continue

            raw = bf.knnMatch(des_prev, des_curr, k=2)
            good = [m for m, n2 in raw if m.distance < 0.75 * n2.distance]
            if len(good) < 20:
                R_all[idx], C_all[idx] = R_all[idx - 1], C_all[idx - 1]
                vo_mode[idx] = "vo_hold"
                continue

            p0 = np.float64([kp_prev[m.queryIdx].pt for m in good])
            p1 = np.float64([kp_curr[m.trainIdx].pt for m in good])

            H, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 2.0)
            pure_rot = False
            R_rot = np.eye(3)
            if H is not None:
                u, _, vt = np.linalg.svd(K_inv @ H @ K)
                R_rot = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
                H_rot = K @ R_rot @ K_inv
                p0_h = np.hstack([p0, np.ones((len(p0), 1))])
                pred = (H_rot @ p0_h.T).T
                pred = pred[:, :2] / pred[:, 2:3]
                if float(np.median(np.linalg.norm(pred - p1, axis=1))) < 2.0:
                    pure_rot = True

            if pure_rot:
                R_all[idx], C_all[idx] = chain_relative_pose(
                    R_all[prev_idx], C_all[prev_idx], R_rot, np.zeros(3), scale=0.0
                )
                vo_mode[idx] = "homography_rot"
                kp_prev, des_prev, prev_idx = kp_curr, des_curr, idx
            else:
                E, mE = cv2.findEssentialMat(p0, p1, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0)
                if E is None:
                    R_all[idx], C_all[idx] = R_all[idx - 1], C_all[idx - 1]
                    vo_mode[idx] = "vo_hold"
                    continue
                n_in, R_rel, t_rel, m_pose = cv2.recoverPose(E, p0, p1, K, mask=mE)
                if n_in < 15:
                    R_all[idx], C_all[idx] = R_all[idx - 1], C_all[idx - 1]
                    vo_mode[idx] = "vo_hold"
                    continue
                R_all[idx], C_all[idx] = chain_relative_pose(
                    R_all[prev_idx], C_all[prev_idx], R_rel, t_rel.ravel(), scale=1.0
                )
                inl = m_pose.ravel() > 0
                P0_m = K @ np.hstack([R_all[prev_idx].T, (-R_all[prev_idx].T @ C_all[prev_idx])[:, None]])
                P1_m = K @ np.hstack([R_all[idx].T, (-R_all[idx].T @ C_all[idx])[:, None]])
                X4 = cv2.triangulatePoints(P0_m, P1_m, p0[inl].T, p1[inl].T)
                pts_cloud.append((X4[:3] / X4[3:4]).T)
                kp_prev, des_prev, prev_idx = kp_curr, des_curr, idx

        cap.release()
        pts_w = np.vstack(pts_cloud) if pts_cloud else np.zeros((0, 3), dtype=np.float64)
        return R_all, C_all, pts_w, pose_valid, vo_mode


_PLY_DTYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _project_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    return u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt


def _read_ply_xyz(ply_path: Path, max_points: int = 50000) -> np.ndarray:
    """Parse ASCII or binary PLY vertex XYZ coordinates into an (N, 3) float64 array."""
    fmt: str | None = None
    n_verts = 0
    in_vertex = False
    props: list[tuple[str, str]] = []

    with ply_path.open("rb") as f:
        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            line = raw_line.decode("ascii", errors="ignore").strip()
            if not line or line.startswith("comment"):
                continue
            parts = line.split()
            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_verts = int(parts[2])
            elif parts[0] == "property" and in_vertex and len(parts) >= 3 and parts[1] != "list":
                code = _PLY_DTYPES.get(parts[1])
                if code is None:
                    raise ValueError(f"Unsupported PLY property type {parts[1]!r} in {ply_path}")
                props.append((parts[2], code))
            elif parts[0] == "end_header":
                break

        if n_verts <= 0 or not props:
            return np.zeros((0, 3), dtype=np.float64)

        names = [name for name, _ in props]
        if not all(ax in names for ax in ("x", "y", "z")):
            raise ValueError(f"PLY vertex schema missing x/y/z coordinates: {names}")

        if fmt == "ascii":
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            pts_list = []
            for _ in range(n_verts):
                row = f.readline().decode("ascii", errors="ignore").split()
                if len(row) >= len(names):
                    pts_list.append([float(row[ix]), float(row[iy]), float(row[iz])])
            pts = np.asarray(pts_list, dtype=np.float64) if pts_list else np.zeros((0, 3), dtype=np.float64)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(name, endian + code) for name, code in props])
            recs = np.fromfile(f, dtype=dt, count=n_verts)
            pts = np.column_stack([recs["x"], recs["y"], recs["z"]]).astype(np.float64)
        else:
            raise ValueError(f"Unsupported PLY format {fmt!r} in {ply_path}")

    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


def _load_da3_camera_poses(poses_path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 4x4, 3x4, or TUM quaternion poses from camera_poses.txt into SO(3) + translation."""
    lines = [
        ln.strip()
        for ln in poses_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    R_raw = np.zeros((n_frames, 3, 3), dtype=np.float64)
    t_raw = np.zeros((n_frames, 3), dtype=np.float64)
    valid = np.zeros(n_frames, dtype=bool)

    for idx, ln in enumerate(lines[:n_frames]):
        vals = np.asarray([float(x) for x in ln.split()], dtype=np.float64)
        if len(vals) == 16:
            mat = vals.reshape(4, 4)
            R, t = mat[:3, :3], mat[:3, 3]
        elif len(vals) == 12:
            mat = vals.reshape(3, 4)
            R, t = mat[:3, :3], mat[:3, 3]
        elif len(vals) in (7, 8):
            tx, ty, tz, qx, qy, qz, qw = vals[-7:]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz], dtype=np.float64)
        else:
            raise ValueError(f"Expected 7, 8, 12, or 16 floats at line {idx + 1} in {poses_path}; got {len(vals)}")
        if np.all(np.isfinite(R)) and np.all(np.isfinite(t)):
            R_raw[idx] = _project_so3(R)
            t_raw[idx] = t
            valid[idx] = True
    return R_raw, t_raw, valid


class Da3EgoMotion(BaseEgoMotion):
    """Optional cloud GPU adapter loading DA3-Streaming or MegaSaM poses and point cloud."""

    def __init__(self, subdir: str = "da3", convention: str = "auto") -> None:
        self.subdir = subdir
        self.convention = convention

    def _find_asset(self, cache_dir: Path, rel_paths: list[Path]) -> Path | None:
        for rp in rel_paths:
            cand = cache_dir / rp
            if cand.exists():
                return cand
        return None

    def estimate(
        self,
        video_path: Path,
        s1: Stage1Result,
        cache_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_frames = len(s1.frame_idx)
        subdirs = [self.subdir] if self.subdir == "da3" else [self.subdir, "da3"]
        poses_file = self._find_asset(
            cache_dir, [*(Path(sd) / "camera_poses.txt" for sd in subdirs), Path("camera_poses.txt")]
        )
        pcd_file = self._find_asset(
            cache_dir,
            [
                *(Path(sd) / "pcd" / "combined_pcd.ply" for sd in subdirs),
                *(Path(sd) / "combined_pcd.ply" for sd in subdirs),
                Path("pcd") / "combined_pcd.ply",
                Path("combined_pcd.ply"),
            ],
        )
        if poses_file is None or pcd_file is None:
            raise RuntimeError(
                f"DA3/MegaSaM outputs not found in '{cache_dir / self.subdir}'. Export frames with "
                "`python run_pipeline.py --profile gpu --export-frames`, run `da3_streaming.py`, and copy "
                "`camera_poses.txt` + `pcd/combined_pcd.ply` into that directory."
            )

        R_raw, t_raw, pose_valid = _load_da3_camera_poses(poses_file, n_frames)
        valid_idx = np.where(pose_valid)[0]
        if len(valid_idx) == 0:
            raise RuntimeError(f"No valid camera poses parsed from '{poses_file}'.")

        pts_w = _read_ply_xyz(pcd_file, max_points=50000)

        # Resolve c2w vs w2c extrinsic convention via frustum cheirality against the 3D cloud.
        if self.convention == "w2c":
            use_w2c = True
        elif self.convention == "c2w":
            use_w2c = False
        else:
            use_w2c = False
            if len(pts_w) > 0:
                K = CONFIG.camera.K
                h, w = CONFIG.camera.height, CONFIG.camera.width
                samp_pts = pts_w[:: max(1, len(pts_w) // 4000)][:4000]
                check_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, min(10, len(valid_idx)), dtype=int)]

                def _frustum_frac(Rw: np.ndarray, Cw: np.ndarray) -> float:
                    pc = (Rw.T @ (samp_pts - Cw).T).T
                    in_z = pc[:, 2] > 0.1
                    if not np.any(in_z):
                        return 0.0
                    proj = (K @ pc[in_z].T).T
                    u = proj[:, 0] / proj[:, 2]
                    v = proj[:, 1] / proj[:, 2]
                    ok_uv = (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
                    return float(np.sum(ok_uv)) / len(samp_pts)

                frac_c2w = [_frustum_frac(R_raw[f], t_raw[f]) for f in check_idx]
                frac_w2c = [_frustum_frac(R_raw[f].T, -R_raw[f].T @ t_raw[f]) for f in check_idx]
                use_w2c = float(np.median(frac_w2c)) > float(np.median(frac_c2w)) + 0.10

        poses_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        vo_mode = np.full(n_frames, "none", dtype="U16")
        mode_tag = "da3_streaming" if self.subdir == "da3" else self.subdir
        for f in valid_idx:
            if use_w2c:
                R_wc = R_raw[f].T
                C_w = -R_wc @ t_raw[f]
            else:
                R_wc = R_raw[f]
                C_w = t_raw[f]
            poses_map[int(f)] = (R_wc, C_w)
            vo_mode[f] = mode_tag

        R_all, C_all = _interpolate_poses(poses_map, n_frames, vo_mode, pose_valid)

        # Strip dynamic foreground points (person + ball) using Stage 1 static masks.
        if len(pts_w) > 0:
            K = CONFIG.camera.K
            h, w = CONFIG.camera.height, CONFIG.camera.width
            vis_cnt = np.zeros(len(pts_w), dtype=np.int32)
            dyn_cnt = np.zeros(len(pts_w), dtype=np.int32)
            mask_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, min(12, len(valid_idx)), dtype=int)]
            for f in mask_idx:
                sm = Stage1Result.unpack_static_mask(s1.static_masks_packed[f])
                pc = (R_all[f].T @ (pts_w - C_all[f]).T).T
                in_z = (pc[:, 2] > 0.2) & (pc[:, 2] < 40.0)
                if not np.any(in_z):
                    continue
                proj = (K @ pc[in_z].T).T
                u = np.round(proj[:, 0] / proj[:, 2]).astype(int)
                v = np.round(proj[:, 1] / proj[:, 2]).astype(int)
                in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                vis_ids = np.where(in_z)[0][in_img]
                vis_cnt[vis_ids] += 1
                is_dyn = ~sm[v[in_img], u[in_img]]
                dyn_cnt[vis_ids[is_dyn]] += 1
            keep = (vis_cnt == 0) | ((dyn_cnt / np.maximum(vis_cnt, 1)) < 0.5)
            if int(np.sum(keep)) >= max(20, int(0.25 * len(pts_w))):
                pts_w = pts_w[keep]

        return R_all, C_all, pts_w, pose_valid, vo_mode


def create_ego_motion(name: str) -> BaseEgoMotion:
    """Instantiate the requested Stage 3 ego-motion backend."""
    key = name.lower()
    if key == "pycolmap":
        return PyColmapEgoMotion()
    if key == "sift":
        return SiftVoEgoMotion()
    if key in ("da3", "megasam"):
        return Da3EgoMotion(subdir=key)
    raise ValueError(f"Unknown VO backend: {name!r}")


def save_stage3_diagnostics(
    s3: Stage3Result,
    s1: Stage1Result,
    df2: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Plot top-down 2D terrace map, elevation stability, and pre-kick camera-independence proof."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = len(s3.pose_valid)
    valid_ball = s1.detected & df2["valid"].to_numpy() & s3.pose_valid

    P_g = np.full((n, 3), np.nan, dtype=np.float64)
    for k in np.where(valid_ball)[0]:
        Pc = df2.loc[k, ["X_c", "Y_c", "Z_c"]].to_numpy(dtype=np.float64)
        P_g[k] = s3.R_gc[k] @ Pc + s3.C_g[k]

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_z = fig.add_subplot(gs[0, 1])
    ax_stab = fig.add_subplot(gs[1, 1])

    pts = s3.sparse_points_g
    floor_vis = (np.abs(pts[:, 2]) < 1.3) & (pts[:, 0] > -2.0) & (pts[:, 0] < 12.0) & (np.abs(pts[:, 1]) < 6.0)
    if np.any(floor_vis):
        ax_map.scatter(pts[floor_vis, 0], pts[floor_vis, 1], s=2.5, c="lightgray", alpha=0.65, label="Terrace SfM points")

    ax_map.plot(s3.C_g[:, 0], s3.C_g[:, 1], color="tab:blue", lw=1.8, label="Camera path")
    for k in range(0, n, 40):
        fwd = s3.R_gc[k, :2, 2]
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-6) * 0.45
        ax_map.arrow(s3.C_g[k, 0], s3.C_g[k, 1], fwd[0], fwd[1], head_width=0.10, color="tab:blue", alpha=0.8)

    s0, s1_end = CONFIG.video.stationary_frames
    k1_0, k1_1 = CONFIG.video.kick1_frames
    k2_0, k2_1 = CONFIG.video.kick2_frames

    stat_idx = np.arange(s0, min(s1_end + 1, n))
    k1_idx = np.arange(k1_0, min(k1_1 + 1, n))
    k2_idx = np.arange(k2_0, min(k2_1 + 1, n))

    if len(stat_idx):
        ax_map.scatter(P_g[stat_idx, 0], P_g[stat_idx, 1], s=14, color="tab:green", alpha=0.75, label="Ball stationary (F0-158)")
    if np.all(np.isfinite(s3.stat_ball_tri_g)):
        ax_map.scatter([s3.stat_ball_tri_g[0]], [s3.stat_ball_tri_g[1]], marker="*", s=160, color="darkgreen", zorder=6, label="Triangulated anchor")
    if len(k1_idx):
        ax_map.plot(P_g[k1_idx, 0], P_g[k1_idx, 1], color="tab:orange", lw=1.8, label="Raw Kick 1 (F161-206)")
    if len(k2_idx):
        ax_map.plot(P_g[k2_idx, 0], P_g[k2_idx, 1], color="tab:red", lw=1.8, label="Raw Kick 2 (F379-461)")
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X_g forward (m)")
    ax_map.set_ylabel("Y_g left (m)")
    ax_map.set_title("Stage 3: Ground-Aligned 2D Terrace Map")
    ax_map.legend(loc="best", fontsize=8.5)
    ax_map.grid(alpha=0.3)

    frames = np.arange(n)
    ax_z.plot(frames, s3.C_g[:, 2], color="tab:blue", lw=1.4, label="Camera height C_z (m)")
    ax_z.plot(frames, P_g[:, 2], color="tab:orange", lw=1.4, label="Ball Z_g (m)")
    ax_z.axhline(CONFIG.ball.radius_m, color="black", ls="--", lw=1.1, label="True ball radius (0.11 m)")
    ax_z.axvspan(*CONFIG.video.gap_frames, color="gray", alpha=0.12)
    ax_z.set_ylabel("Elevation Z_g (m)")
    ax_z.set_title("Ground-Plane Alignment Check")
    ax_z.legend(loc="center right", fontsize=8.5)
    ax_z.grid(alpha=0.3)

    if len(stat_idx) and np.any(np.isfinite(P_g[stat_idx, 1])):
        xc_stat = (df2.loc[stat_idx, "X_c"].to_numpy() - df2.loc[stat_idx, "X_c"].median()) * 100.0
        yg_stat = (P_g[stat_idx, 1] - np.nanmedian(P_g[stat_idx, 1])) * 100.0
        ax_stab.plot(stat_idx, xc_stat, color="tab:purple", lw=1.4, label="Raw camera-frame X_c offset (cm)")
        ax_stab.plot(stat_idx, yg_stat, color="tab:green", lw=1.6, label="World-frame lateral Y_g residual (cm)")
        ax_stab.axhline(0.0, color="gray", ls=":", lw=0.9)
        ax_stab.legend(loc="upper left", fontsize=8.5)
    ax_stab.set_xlabel("Frame Index (Stationary Window)")
    ax_stab.set_ylabel("Lateral Drift (cm)")
    ax_stab.set_title("Camera-Motion Decoupling Verification")
    ax_stab.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "stage3_ego_diagnostics.png", dpi=150)
    plt.close(fig)


def run_stage3(profile_name: str = "local", vo_name: str | None = None) -> Stage3Result:
    """Execute Stage 3 ego-motion recovery, metric scale anchoring, and ground alignment."""
    CONFIG.paths.ensure_dirs(profile_name)
    cache_dir = CONFIG.paths.cache_dir(profile_name)
    profile = CONFIG.get_profile(profile_name)

    s1 = Stage1Result.load(cache_dir)
    df2 = pd.read_csv(cache_dir / "stage2_3d.csv")

    backend = create_ego_motion(vo_name or profile.vo)
    R_wc_u, C_w_u, pts_w_u, pose_valid, vo_mode = backend.estimate(CONFIG.paths.video, s1, cache_dir)

    alpha, stat_ball_w, tri_err_px = anchor_metric_scale(
        R_wc_u,
        C_w_u,
        pose_valid,
        s1,
        df2,
        CONFIG.camera.K,
        stat_range=None,
        points_w_unscaled=pts_w_u,
    )

    C_w = alpha * C_w_u
    pts_w = alpha * pts_w_u
    n_up, d_plane, _ = fit_ground_plane_ransac(pts_w, C_w[0], R_wc_u[0])
    R_gw, t_gw = build_ground_alignment(n_up, d_plane, R_wc_u[0], C_w[0])

    R_gc = np.einsum("ij,njk->nik", R_gw, R_wc_u)
    C_g = (R_gw @ C_w.T).T + t_gw
    pts_g = ((R_gw @ pts_w.T).T + t_gw).astype(np.float32)
    stat_ball_g = (R_gw @ stat_ball_w + t_gw) if np.all(np.isfinite(stat_ball_w)) else np.full(3, np.nan, dtype=np.float64)

    result = Stage3Result(
        R_wc=R_wc_u,
        C_w=C_w,
        R_gw=R_gw,
        t_gw=t_gw,
        R_gc=R_gc,
        C_g=C_g,
        scale_alpha=alpha,
        ground_plane_abcd=np.array([*n_up, d_plane], dtype=np.float64),
        sparse_points_g=pts_g,
        stat_ball_tri_g=stat_ball_g,
        pose_valid=pose_valid,
        vo_mode=vo_mode,
    )
    result.save(cache_dir)
    save_stage3_diagnostics(result, s1, df2, CONFIG.paths.plots_dir)
    max_jump = float(np.max(np.linalg.norm(np.diff(C_g, axis=0), axis=1))) if len(C_g) >= 2 else 0.0
    if np.isfinite(tri_err_px):
        print(
            f"Triangulated stationary ball reproj error: {tri_err_px:.2f} px | height Z_g={stat_ball_g[2]:.3f} m | "
            f"cam Z_g=[{C_g[:, 2].min():.3f}, {C_g[:, 2].max():.3f}] m, max jump={max_jump:.3f} m"
        )
    else:
        print(
            f"Floor-contact metric scale anchored (alpha={alpha:.5f}) | "
            f"cam Z_g=[{C_g[:, 2].min():.3f}, {C_g[:, 2].max():.3f}] m, max jump={max_jump:.3f} m"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Camera ego-motion, metric scale & ground alignment.")
    parser.add_argument("--profile", default="local", choices=("local", "gpu"))
    parser.add_argument("--vo", default=None, choices=("pycolmap", "sift", "da3", "megasam"))
    args = parser.parse_args()

    s3 = run_stage3(args.profile, args.vo)
    cache_dir = CONFIG.paths.cache_dir(args.profile)
    s1 = Stage1Result.load(cache_dir)
    df2 = pd.read_csv(cache_dir / "stage2_3d.csv")

    print(f"Localized frames: {int(s3.pose_valid.sum())}/{len(s3.pose_valid)} | scale alpha={s3.scale_alpha:.5f}")
    unique, counts = np.unique(s3.vo_mode, return_counts=True)
    print("VO modes:", dict(zip(unique.tolist(), counts.tolist())))
    print(f"Camera height Z_g: mean={s3.C_g[:, 2].mean():.3f} m (min={s3.C_g[:, 2].min():.3f}, max={s3.C_g[:, 2].max():.3f})")
    print(f"Metric camera span: {np.linalg.norm(s3.C_g.max(axis=0) - s3.C_g.min(axis=0)):.3f} m")

    s0, s1_end = CONFIG.video.stationary_frames
    clean_stat = [
        k for k in range(s0, s1_end + 1)
        if s1.detected[k] and df2.loc[k, "method"] not in ("occluded_deweighted", "clipped_deweighted")
    ]
    P_g_clean = np.array([s3.R_gc[k] @ df2.loc[k, ["X_c", "Y_c", "Z_c"]].to_numpy(float) + s3.C_g[k] for k in clean_stat])
    P_c_clean = df2.loc[clean_stat, ["X_c", "Y_c", "Z_c"]].to_numpy(float)
    print(
        f"Clean stationary ({len(clean_stat)}f): raw cam XY std={(P_c_clean[:, :2].std(axis=0)*100).round(2)} cm | "
        f"world XY std={(P_g_clean[:, :2].std(axis=0)*100).round(2)} cm | Z_g med={np.median(P_g_clean[:, 2]):.3f} m"
    )


if __name__ == "__main__":
    main()
