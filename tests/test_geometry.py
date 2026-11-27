# Unit tests for sphere 3D math, camera pose chaining, ground alignment, scale recovery, and IMM+RTS tracking.

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import BallConfig, CameraConfig, CONFIG
from src.detector_2d import Stage1Result
from src.ego_motion import (
    Stage3Result,
    anchor_metric_scale,
    build_ground_alignment,
    chain_relative_pose,
    fit_ground_plane_ransac,
    select_keyframes_optical_flow,
    solve_scale_soft_l1,
    triangulate_stationary_ball,
)
from src.localizer_3d import fuse_sphere_with_ground_plane, intersect_ray_with_plane_z, localize_sphere_3d, localize_stage1
from src.tracker_3d import IMMTracker3D, run_imm_rts_segment, step_airborne, step_rolling


def _project_sphere_horizon_rays(
    P_c: np.ndarray,
    radius: float,
    camera: CameraConfig,
    n_rays: int = 720,
) -> tuple[float, float, float, float, float, float]:
    """Analytically project the exact tangent-cone horizon circle of a 3D sphere onto the sensor."""
    d = float(np.linalg.norm(P_c))
    u_axis = P_c / d
    sin_theta = radius / d
    cos_theta = float(np.sqrt(1.0 - sin_theta**2))

    # Build orthonormal basis orthogonal to cone axis u_axis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(u_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u_axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_axis, e1)

    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    horizon_pts = (
        P_c[None, :]
        - radius * sin_theta * u_axis[None, :]
        + radius * cos_theta * (np.cos(angles)[:, None] * e1[None, :] + np.sin(angles)[:, None] * e2[None, :])
    )

    uv = (camera.K @ horizon_pts.T).T
    uv = (uv[:, :2] / uv[:, 2:3]).astype(np.float32)

    (u, v), (d1, d2), ang = cv2.fitEllipseAMS(uv.reshape(-1, 1, 2))
    a_px = 0.5 * max(d1, d2)
    b_px = 0.5 * min(d1, d2)
    if d1 < d2:
        ang = (ang + 90.0) % 180.0
    area_px = float(cv2.contourArea(uv.reshape(-1, 1, 2)))
    return float(u), float(v), float(a_px), float(b_px), float(area_px), float(ang)


def test_penne_cubic_exact_off_axis_30deg() -> None:
    """Verify Penne cubic eliminates off-axis corner stretch bias to sub-2mm accuracy."""
    P_true = np.array([2.4, 1.5, 5.0], dtype=np.float64)
    ball = CONFIG.ball
    cam = CONFIG.camera
    u, v, a_px, b_px, area_px, ang = _project_sphere_horizon_rays(P_true, ball.radius_m, cam)

    pose = localize_sphere_3d(u, v, a_px, b_px, area_px, camera=cam, ball=ball, ellipse_angle_deg=ang)
    assert pose.valid
    assert pose.method == "penne_cubic"

    P_rec = np.array([pose.X_c, pose.Y_c, pose.Z_c])
    assert np.linalg.norm(P_rec - P_true) < 0.002
    assert abs(pose.distance_m - float(np.linalg.norm(P_true))) < 0.002

    # Confirm naive semi-major pinhole depth suffers >10% underestimation at this off-axis angle.
    Z_naive = cam.f_mean * ball.diameter_m / (2.0 * a_px)
    assert (P_true[2] - Z_naive) / P_true[2] > 0.10


def test_minor_axis_exact_under_motion_blur() -> None:
    """Verify transverse semi-minor axis recovers exact depth despite 50% motion-blur stretch."""
    P_true = np.array([-1.2, 0.8, 4.0], dtype=np.float64)
    ball = CONFIG.ball
    cam = CONFIG.camera
    u, v, a_px, b_px, area_px, ang = _project_sphere_horizon_rays(P_true, ball.radius_m, cam)

    pose = localize_sphere_3d(
        u, v, a_px * 1.5, b_px, area_px * 1.5, camera=cam, ball=ball, is_blurred=True, ellipse_angle_deg=ang
    )
    assert pose.valid
    assert pose.method == "minor_axis"

    P_rec = np.array([pose.X_c, pose.Y_c, pose.Z_c])
    assert np.linalg.norm(P_rec - P_true) < 0.002


def test_heteroscedastic_sigma_and_occlusion_inflation() -> None:
    """Verify sigma_Z scales as Z^2 and inflates 10x under person occlusion."""
    cam = CameraConfig(width=1280, height=720)
    ball = BallConfig()
    b_5m = cam.f_mean * ball.radius_m / 5.0
    b_10m = cam.f_mean * ball.radius_m / 10.0

    p5 = localize_sphere_3d(cam.cx, cam.cy, b_5m, b_5m, np.pi * b_5m**2, camera=cam, ball=ball)
    p10 = localize_sphere_3d(cam.cx, cam.cy, b_10m, b_10m, np.pi * b_10m**2, camera=cam, ball=ball)
    assert abs((p10.sigma_Z / p5.sigma_Z) - 4.0) < 0.05

    p5_occ = localize_sphere_3d(
        cam.cx, cam.cy, b_5m, b_5m, np.pi * b_5m**2, camera=cam, ball=ball, occluded_person=True
    )
    assert p5_occ.method == "occluded_deweighted"
    assert abs((p5_occ.sigma_Z / p5.sigma_Z) - 10.0) < 1e-6


def test_pose_chaining_invariance() -> None:
    """Verify relative pose chaining keeps a stationary 3D world point invariant."""
    rng = np.random.default_rng(7)
    P_world = np.array([1.2, -0.5, 6.5], dtype=np.float64)
    R_wc_true = [np.eye(3)]
    C_true = [np.zeros(3)]

    for step in range(1, 20):
        ang = 0.04 * step
        axis = np.array([0.1, 1.0, -0.2])
        axis /= np.linalg.norm(axis)
        K_ax = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_wc_true.append(np.eye(3) + np.sin(ang) * K_ax + (1.0 - np.cos(ang)) * (K_ax @ K_ax))
        C_true.append(np.array([0.05 * step, -0.02 * np.sin(step), 0.03 * step]) + rng.normal(0, 1e-3, 3))

    R_wc_rec, C_rec = np.eye(3), np.zeros(3)
    for k in range(1, len(C_true)):
        R_rel = R_wc_true[k].T @ R_wc_true[k - 1]
        t_rel_full = R_wc_true[k].T @ (C_true[k - 1] - C_true[k])
        scale = float(np.linalg.norm(t_rel_full))
        t_hat = t_rel_full / scale
        R_wc_rec, C_rec = chain_relative_pose(R_wc_rec, C_rec, R_rel, t_hat, scale=scale)
        P_c_k = R_wc_true[k].T @ (P_world - C_true[k])
        P_w_rec = R_wc_rec @ P_c_k + C_rec
        assert np.linalg.norm(P_w_rec - P_world) < 1e-10


def test_ground_plane_gram_schmidt_alignment() -> None:
    """Verify RANSAC plane fit + Gram-Schmidt maps floor points to Z_g=0 and ball to R=0.11m."""
    rng = np.random.default_rng(12)
    # Construct tilted world plane with normal tilted ~20 deg.
    n_true = np.array([0.05, -0.94, -0.34], dtype=np.float64)
    n_true /= np.linalg.norm(n_true)
    d_true = 1.65
    C_w0 = np.zeros(3, dtype=np.float64)
    R_wc0 = np.eye(3, dtype=np.float64)

    # Sample synthetic floor points inside forward terrace region (Z_c0 in 2..8 m).
    pts_floor = []
    for x_g in np.linspace(-1.5, 1.5, 12):
        for z_g in np.linspace(2.0, 8.0, 12):
            # Solve n_true . [x_g, y, z_g] + d_true = 0 for y.
            y = -(n_true[0] * x_g + n_true[2] * z_g + d_true) / n_true[1]
            pts_floor.append([x_g, y, z_g])
    pts_floor = np.asarray(pts_floor, dtype=np.float64) + rng.normal(0.0, 1e-3, (144, 3))
    pts_wall = np.array([[0.0, -2.0, 8.5], [1.5, -1.8, 8.2], [-1.5, -1.7, 7.9]])
    pts_all = np.vstack([pts_floor, pts_wall])

    n_up, d_plane, _ = fit_ground_plane_ransac(pts_all, C_w0, R_wc0, inlier_thresh_m=0.02)
    R_gw, t_gw = build_ground_alignment(n_up, d_plane, R_wc0, C_w0)

    assert abs(float(np.linalg.det(R_gw)) - 1.0) < 1e-10
    assert np.linalg.norm(R_gw @ R_gw.T - np.eye(3)) < 1e-10

    floor_g = (R_gw @ pts_floor.T).T + t_gw
    assert np.max(np.abs(floor_g[:, 2])) < 0.01

    ball_w = np.array([0.0, -(d_true) / n_true[1], 0.0]) + 0.11 * n_true
    ball_g = R_gw @ ball_w + t_gw
    assert abs(float(ball_g[2]) - 0.11) < 0.01
    cam_g = R_gw @ C_w0 + t_gw
    assert cam_g[2] > 1.5


def test_stationary_ball_triangulation_and_scale() -> None:
    """Verify multi-view DLT triangulation + soft_l1 scale recovery from unknown-scale cameras."""
    cam = CONFIG.camera
    alpha_true = 13.7
    P_ball_metric = np.array([0.5, 1.1, 6.8], dtype=np.float64)

    R_cw_list, t_cw_list, uv_list, C_u_list, dist_m = [], [], [], [], []
    for k in range(15):
        ang = 0.03 * (k - 7)
        R_cw = np.array([[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]])
        C_metric = np.array([0.12 * (k - 7), 0.05 * np.sin(k), 0.06 * k], dtype=np.float64)
        C_u = C_metric / alpha_true
        t_cw_u = -R_cw @ C_u

        P_c_m = R_cw @ (P_ball_metric - C_metric)
        proj = cam.K @ P_c_m
        uv = proj[:2] / proj[2]

        R_cw_list.append(R_cw)
        t_cw_list.append(t_cw_u)
        uv_list.append(uv)
        C_u_list.append(C_u)
        dist_m.append(float(np.linalg.norm(P_c_m)))

    X_tri, errs = triangulate_stationary_ball(R_cw_list, t_cw_list, uv_list, cam.K)
    assert float(np.max(errs)) < 1e-6

    alpha_rec = solve_scale_soft_l1(X_tri, np.asarray(C_u_list), np.asarray(dist_m))
    assert abs(alpha_rec - alpha_true) < 1e-5
    assert np.linalg.norm(alpha_rec * X_tri - P_ball_metric) < 1e-4


def test_ray_ground_intersection_and_airborne_detection() -> None:
    """Verify ray-ground intersection hits Z_g=R exactly and detects sustained airborne elevation."""
    cam = CONFIG.camera
    ball = CONFIG.ball
    n_frames = 6

    # Build camera looking forward/down at floor from height 1.7m (world ground frame = identity).
    C_g_0 = np.array([0.0, 0.0, 1.7], dtype=np.float64)
    # Camera axes in ground frame: +X_c -> -Y_g, +Y_c -> tilted down, +Z_c -> tilted forward-down.
    ez_c = np.array([1.0, 0.0, -0.35], dtype=np.float64)
    ez_c /= np.linalg.norm(ez_c)
    ex_c = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    ey_c = np.cross(ez_c, ex_c)
    R_gc_0 = np.column_stack([ex_c, ey_c, ez_c])

    P_true = np.vstack([
        np.tile([4.0, -0.5, ball.radius_m], (3, 1)),
        np.tile([4.0, -0.5, 0.80], (3, 1)),
    ])

    hit_ok, p_hit, _ = intersect_ray_with_plane_z(C_g_0, P_true[0] - C_g_0, ball.radius_m)
    assert hit_ok
    assert np.linalg.norm(p_hit - P_true[0]) < 1e-10

    u_arr, v_arr, a_arr, b_arr, area_arr, ang_arr = [], [], [], [], [], []
    for k in range(n_frames):
        Pc = R_gc_0.T @ (P_true[k] - C_g_0)
        u, v, a, b, area, ang = _project_sphere_horizon_rays(Pc, ball.radius_m, cam)
        u_arr.append(u); v_arr.append(v); a_arr.append(a); b_arr.append(b); area_arr.append(area); ang_arr.append(ang)

    s1 = Stage1Result(
        frame_idx=np.arange(n_frames, dtype=np.int32),
        timestamp_s=np.arange(n_frames, dtype=np.float64) / 30.0,
        detected=np.ones(n_frames, dtype=bool),
        det_source=np.full(n_frames, "yolo_tile", dtype=str),
        bbox_xyxy=np.zeros((n_frames, 4), dtype=np.float32),
        center_uv=np.column_stack([u_arr, v_arr]),
        semi_axes_ab=np.column_stack([a_arr, b_arr]),
        ellipse_angle_deg=np.asarray(ang_arr, dtype=np.float64),
        mask_area_px=np.asarray(area_arr, dtype=np.float64),
        conf=np.ones(n_frames, dtype=np.float32),
        is_blurred=np.zeros(n_frames, dtype=bool),
        occluded_person=np.zeros(n_frames, dtype=bool),
        border_clipped=np.zeros(n_frames, dtype=bool),
        static_masks_packed=np.zeros((n_frames, 1), dtype=np.uint8),
    )
    df2 = localize_stage1(s1, cam, ball)
    s3 = Stage3Result(
        R_wc=np.tile(R_gc_0, (n_frames, 1, 1)),
        C_w=np.tile(C_g_0, (n_frames, 1)),
        R_gw=np.eye(3),
        t_gw=np.zeros(3),
        R_gc=np.tile(R_gc_0, (n_frames, 1, 1)),
        C_g=np.tile(C_g_0, (n_frames, 1)),
        scale_alpha=1.0,
        ground_plane_abcd=np.array([0.0, 0.0, 1.0, 0.0]),
        sparse_points_g=np.zeros((0, 3), dtype=np.float32),
        stat_ball_tri_g=P_true[0],
        pose_valid=np.ones(n_frames, dtype=bool),
        vo_mode=np.full(n_frames, "sfm_kf", dtype=str),
    )

    df_fused, _ = fuse_sphere_with_ground_plane(s1, df2, s3, cam, ball)
    assert not bool(df_fused.loc[0, "is_airborne"])
    assert not bool(df_fused.loc[2, "is_airborne"])
    assert np.all(df_fused.loc[3:5, "is_airborne"].to_numpy())
    assert abs(float(df_fused.loc[4, "Z_g_fused"]) - 0.80) < 0.01


def test_imm_rts_recovers_rolling_and_ballistic_physics() -> None:
    """Verify 3-mode IMM+RTS identifies stationary/rolling/airborne modes and reduces RMSE >2x."""
    rng = np.random.default_rng(99)
    dt = 1.0 / 30.0
    ball = CONFIG.ball

    # 1. Rolling trajectory (60 frames) with anisotropic depth noise.
    n_roll = 60
    true_roll = np.zeros((n_roll, 6), dtype=np.float64)
    true_roll[0] = [1.0, 0.0, ball.radius_m, 3.0, -0.8, 0.0]
    for k in range(1, n_roll):
        true_roll[k] = step_rolling(true_roll[k - 1], dt, ball.rolling_friction, 0.0, ball)
    R_roll = np.tile(np.diag([0.20**2, 0.05**2, 0.03**2]), (n_roll, 1, 1))
    meas_roll = true_roll[:, :3] + rng.normal(0.0, [0.20, 0.05, 0.03], size=(n_roll, 3))

    _, _, xs_roll, _, mus_roll, _ = run_imm_rts_segment(list(range(n_roll)), meas_roll, R_roll, dt, ball)
    raw_rmse_r = float(np.sqrt(np.mean(np.sum((meas_roll - true_roll[:, :3]) ** 2, axis=1))))
    sm_rmse_r = float(np.sqrt(np.mean(np.sum((xs_roll[:, :3] - true_roll[:, :3]) ** 2, axis=1))))
    assert sm_rmse_r < 0.5 * raw_rmse_r
    assert float(np.mean(mus_roll[-20:, 1])) > 0.70

    # 2. Stationary trajectory (40 frames).
    n_stat = 40
    true_stat = np.tile([5.0, -0.5, ball.radius_m], (n_stat, 1))
    R_stat = np.tile(np.diag([0.12**2, 0.04**2, 0.03**2]), (n_stat, 1, 1))
    meas_stat = true_stat + rng.normal(0.0, [0.12, 0.04, 0.03], size=(n_stat, 3))
    _, _, _, _, mus_stat, _ = run_imm_rts_segment(list(range(n_stat)), meas_stat, R_stat, dt, ball)
    assert float(np.mean(mus_stat[-15:, 0])) > 0.70

    # 3. Airborne ballistic trajectory (35 frames) + direct IMMTracker3D check.
    n_air = 35
    true_air = np.zeros((n_air, 6), dtype=np.float64)
    true_air[0] = [0.5, 0.0, 0.3, 5.0, 0.2, 6.0]
    for k in range(1, n_air):
        true_air[k] = step_airborne(true_air[k - 1], dt, None, ball)
    R_air = np.tile(np.diag([0.22**2, 0.06**2, 0.08**2]), (n_air, 1, 1))
    meas_air = true_air[:, :3] + rng.normal(0.0, [0.22, 0.06, 0.08], size=(n_air, 3))
    _, _, xs_air, _, mus_air, _ = run_imm_rts_segment(list(range(n_air)), meas_air, R_air, dt, ball)
    raw_rmse_a = float(np.sqrt(np.mean(np.sum((meas_air - true_air[:, :3]) ** 2, axis=1))))
    sm_rmse_a = float(np.sqrt(np.mean(np.sum((xs_air[:, :3] - true_air[:, :3]) ** 2, axis=1))))
    assert sm_rmse_a < 0.5 * raw_rmse_a
    assert float(np.mean(mus_air[-15:, 2])) > 0.70

    imm = IMMTracker3D(dt=dt, ball=ball)
    xs, Ps, mu = imm.init_state(true_air[0], np.eye(6) * 0.1)
    out = imm.step(xs, Ps, mu, z=meas_air[1], R_meas=R_air[1])
    assert abs(float(np.sum(out["mu"])) - 1.0) < 1e-6


def _build_synthetic_ball_sequence(
    n_frames: int,
    ball_pos_g: np.ndarray,
    R_gc: np.ndarray,
    C_g: np.ndarray,
    cam: CameraConfig,
    ball: BallConfig,
) -> tuple[Stage1Result, pd.DataFrame]:
    rows = []
    uv_arr = np.zeros((n_frames, 2), dtype=np.float64)
    ab_arr = np.zeros((n_frames, 2), dtype=np.float64)
    static_pack = np.packbits(np.ones(cam.height * cam.width, dtype=bool))

    for k in range(n_frames):
        Pc = R_gc[k].T @ (ball_pos_g[k] - C_g[k])
        u, v, a, b, _, _ = _project_sphere_horizon_rays(Pc, ball.radius_m, cam)
        uv_arr[k] = [u, v]
        ab_arr[k] = [a, b]
        dist_m = float(np.linalg.norm(Pc))
        sig_z = (Pc[2] ** 2 / (cam.f_mean * ball.radius_m)) * 0.5
        rows.append(
            {
                "frame_idx": k,
                "timestamp_s": k / 30.0,
                "valid": True,
                "method": "penne_cubic",
                "X_c": Pc[0],
                "Y_c": Pc[1],
                "Z_c": Pc[2],
                "distance_m": dist_m,
                "sigma_X": dist_m / cam.f_mean,
                "sigma_Y": dist_m / cam.f_mean,
                "sigma_Z": sig_z,
                "eccentricity_shift_px": 0.0,
            }
        )

    s1 = Stage1Result(
        frame_idx=np.arange(n_frames, dtype=np.int32),
        timestamp_s=np.arange(n_frames, dtype=np.float64) / 30.0,
        detected=np.ones(n_frames, dtype=bool),
        det_source=np.full(n_frames, "yolo_tile", dtype=str),
        bbox_xyxy=np.zeros((n_frames, 4), dtype=np.float32),
        center_uv=uv_arr,
        semi_axes_ab=ab_arr,
        ellipse_angle_deg=np.zeros(n_frames, dtype=np.float64),
        mask_area_px=np.pi * ab_arr[:, 0] * ab_arr[:, 1],
        conf=np.ones(n_frames, dtype=np.float32),
        is_blurred=np.zeros(n_frames, dtype=bool),
        occluded_person=np.zeros(n_frames, dtype=bool),
        border_clipped=np.zeros(n_frames, dtype=bool),
        static_masks_packed=np.vstack([static_pack for _ in range(n_frames)]),
    )
    return s1, pd.DataFrame(rows)


def test_auto_stationary_ball_scale_at_end_of_sequence() -> None:
    """Verify Tier A finds a late stationary segment automatically without hardcoded frame ranges."""
    cam, ball = CONFIG.camera, CONFIG.ball
    n = 80
    alpha_true = 14.2

    ez_c = np.array([1.0, 0.0, -0.30], dtype=np.float64)
    ez_c /= np.linalg.norm(ez_c)
    ex_c = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    ey_c = np.cross(ez_c, ex_c)
    R_0 = np.column_stack([ex_c, ey_c, ez_c])

    R_wc = np.zeros((n, 3, 3), dtype=np.float64)
    C_metric = np.zeros((n, 3), dtype=np.float64)
    P_ball = np.zeros((n, 3), dtype=np.float64)
    p_late = np.array([5.4, -0.35, ball.radius_m], dtype=np.float64)

    for k in range(n):
        ang = 0.005 * np.sin(0.08 * k)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
        R_wc[k] = Rz @ R_0
        C_metric[k] = [0.02 * k * 0.2, -0.4 * np.sin(0.06 * k), 1.65 + 0.03 * np.cos(0.05 * k)]
        if k < 40:
            P_ball[k] = [2.2 + 0.08 * k, 0.6 - 0.024 * k, ball.radius_m]
        else:
            P_ball[k] = p_late

    s1, df2 = _build_synthetic_ball_sequence(n, P_ball, R_wc, C_metric, cam, ball)
    C_u = C_metric / alpha_true
    alpha_rec, ball_rec, med_err = anchor_metric_scale(
        R_wc, C_u, np.ones(n, dtype=bool), s1, df2, cam.K, stat_range=None
    )
    assert med_err < 0.5
    assert abs(alpha_rec - alpha_true) / alpha_true < 0.01
    assert float(np.linalg.norm(ball_rec - p_late)) < 0.05


def test_moving_ball_floor_contact_scale_fallback() -> None:
    """Verify Tier B recovers metric scale from rolling floor-contact rays when the ball never stops."""
    cam, ball = CONFIG.camera, CONFIG.ball
    n = 50
    alpha_true = 9.5

    ez_c = np.array([1.0, 0.0, -0.30], dtype=np.float64)
    ez_c /= np.linalg.norm(ez_c)
    ex_c = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    ey_c = np.cross(ez_c, ex_c)
    R_0 = np.column_stack([ex_c, ey_c, ez_c])

    R_wc = np.tile(R_0, (n, 1, 1))
    C_metric = np.array([[0.01 * k, -0.5 * np.sin(0.08 * k), 1.68] for k in range(n)], dtype=np.float64)
    P_ball = np.array([[6.5 - 0.09 * k, -0.7 + 0.025 * k, ball.radius_m] for k in range(n)], dtype=np.float64)

    rng = np.random.default_rng(17)
    pts_floor = np.column_stack([rng.uniform(1.5, 8.5, 160), rng.uniform(-2.5, 2.5, 160), np.zeros(160)])

    s1, df2 = _build_synthetic_ball_sequence(n, P_ball, R_wc, C_metric, cam, ball)
    C_u = C_metric / alpha_true
    pts_u = pts_floor / alpha_true
    alpha_rec, ball_tri, _ = anchor_metric_scale(
        R_wc, C_u, np.ones(n, dtype=bool), s1, df2, cam.K, stat_range=None, points_w_unscaled=pts_u
    )
    assert np.all(np.isnan(ball_tri))
    assert abs(alpha_rec - alpha_true) / alpha_true < 0.01


def test_optical_flow_keyframe_selection_adapts_to_motion(tmp_path: Path) -> None:
    """Verify optical-flow keyframes stay sparse during static frames and dense during camera pan."""
    cam = CONFIG.camera
    rng = np.random.default_rng(31)
    canvas = np.full((cam.height, cam.width + 320, 3), 60, dtype=np.uint8)
    for _ in range(350):
        cx = int(rng.integers(20, canvas.shape[1] - 20))
        cy = int(rng.integers(20, canvas.shape[0] - 20))
        rad = int(rng.integers(6, 18))
        col = tuple(int(c) for c in rng.integers(100, 250, 3))
        cv2.circle(canvas, (cx, cy), rad, col, -1)

    avi_path = tmp_path / "synth_motion.avi"
    writer = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (cam.width, cam.height))
    n = 60
    for f in range(n):
        x_off = 0 if f < 30 else 8 * (f - 29)
        frame = canvas[:, x_off : x_off + cam.width]
        writer.write(frame)
    writer.release()

    static_pack = np.packbits(np.ones(cam.height * cam.width, dtype=bool))
    s1 = Stage1Result(
        frame_idx=np.arange(n, dtype=np.int32),
        timestamp_s=np.arange(n, dtype=np.float64) / 30.0,
        detected=np.zeros(n, dtype=bool),
        det_source=np.full(n, "none", dtype=str),
        bbox_xyxy=np.full((n, 4), np.nan, dtype=np.float32),
        center_uv=np.full((n, 2), np.nan, dtype=np.float64),
        semi_axes_ab=np.full((n, 2), np.nan, dtype=np.float64),
        ellipse_angle_deg=np.full(n, np.nan, dtype=np.float64),
        mask_area_px=np.full(n, np.nan, dtype=np.float64),
        conf=np.zeros(n, dtype=np.float32),
        is_blurred=np.zeros(n, dtype=bool),
        occluded_person=np.zeros(n, dtype=bool),
        border_clipped=np.zeros(n, dtype=bool),
        static_masks_packed=np.vstack([static_pack for _ in range(n)]),
    )

    kfs = select_keyframes_optical_flow(avi_path, s1)
    assert kfs[0] == 0 and kfs[-1] == n - 1
    kfs_static = [k for k in kfs if 0 < k < 30]
    kfs_moving = [k for k in kfs if 30 <= k < 59]
    assert len(kfs_moving) >= 2 * max(len(kfs_static), 1)
