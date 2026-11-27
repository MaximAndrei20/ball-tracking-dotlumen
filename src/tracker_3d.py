# Stage 4 3D tracking and prediction. Splits trajectories, runs the 3-mode IMM filter and RTS smoother, refines in image space, and predicts motion.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.config import BallConfig, CameraConfig, CONFIG
from src.detector_2d import Stage1Result
from src.ego_motion import Stage3Result
from src.localizer_3d import fuse_sphere_with_ground_plane


def _rk4_step(rhs_fn, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs_fn(x)
    k2 = rhs_fn(x + 0.5 * dt * k1)
    k3 = rhs_fn(x + 0.5 * dt * k2)
    k4 = rhs_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _numerical_jacobian(step_fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    n = len(x)
    f0 = step_fn(x)
    J = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (step_fn(xp) - f0) / eps
    return J


def step_stationary(x: np.ndarray, dt: float, ball: BallConfig = CONFIG.ball) -> np.ndarray:
    """Propagate stationary state while damping residual velocity."""
    out = x.copy()
    out[:3] = x[:3] + 0.25 * dt * x[3:]
    out[2] = 0.85 * x[2] + 0.15 * ball.radius_m
    out[3:] = 0.15 * x[3:]
    return out


def step_rolling(
    x: np.ndarray,
    dt: float,
    mu_r: float = CONFIG.ball.rolling_friction,
    c_lin: float = 0.0,
    ball: BallConfig = CONFIG.ball,
) -> np.ndarray:
    """Propagate ground-rolling state via RK4 with smooth Coulomb + linear drag."""
    g = ball.gravity_mps2

    def _rhs(s: np.ndarray) -> np.ndarray:
        vx, vy = s[3], s[4]
        spd = float(np.hypot(vx, vy))
        # Smooth regularizer near zero avoids derivative sign chatter during RTS/Jacobians.
        inv_spd = 1.0 / float(np.hypot(spd, 0.03))
        ax = -mu_r * g * vx * inv_spd - c_lin * vx
        ay = -mu_r * g * vy * inv_spd - c_lin * vy
        return np.array([vx, vy, 0.0, ax, ay, 0.0], dtype=np.float64)

    nxt = _rk4_step(_rhs, x, dt)
    if float(np.dot(nxt[3:5], x[3:5])) < 0.0:
        nxt[3:5] = 0.0
    nxt[2] = 0.70 * nxt[2] + 0.30 * ball.radius_m
    nxt[5] = 0.20 * nxt[5]
    return nxt


def step_airborne(
    x: np.ndarray,
    dt: float,
    k_d: float | None = None,
    ball: BallConfig = CONFIG.ball,
) -> np.ndarray:
    """Propagate 3D ballistic projectile with gravity and quadratic aerodynamic drag via RK4."""
    if k_d is None:
        area = np.pi * ball.radius_m**2
        k_d = (ball.air_density * ball.drag_coeff * area) / (2.0 * ball.mass_kg)
    g_vec = np.array([0.0, 0.0, -ball.gravity_mps2], dtype=np.float64)

    def _rhs(s: np.ndarray) -> np.ndarray:
        v = s[3:]
        spd = float(np.linalg.norm(v))
        acc = g_vec - k_d * spd * v
        return np.concatenate([v, acc])

    return _rk4_step(_rhs, x, dt)


class IMMTracker3D:
    """3-Mode Interacting Multiple Model filter (Stationary, Ground Rolling, Airborne Ballistic)."""

    H = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)])

    def __init__(self, dt: float, ball: BallConfig = CONFIG.ball) -> None:
        self.dt = dt
        self.ball = ball
        self.Pi = np.array(
            [[0.990, 0.007, 0.003], [0.004, 0.986, 0.010], [0.003, 0.012, 0.985]],
            dtype=np.float64,
        )
        self.Q_modes = [
            np.diag([1e-7, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6]),
            np.diag([1e-3, 1e-3, 1e-5, (3.0 * dt) ** 2, (3.0 * dt) ** 2, 1e-4]),
            np.diag([1.5e-3, 1.5e-3, 1.5e-3, (3.5 * dt) ** 2, (3.5 * dt) ** 2, (3.5 * dt) ** 2]),
        ]
        self.step_fns = [
            lambda s: step_stationary(s, self.dt, self.ball),
            lambda s: step_rolling(s, self.dt, self.ball.rolling_friction, 0.0, self.ball),
            lambda s: step_airborne(s, self.dt, None, self.ball),
        ]

    def init_state(self, x0: np.ndarray, P0: np.ndarray, mu0: np.ndarray | None = None) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        xs = [x0.copy() for _ in range(3)]
        Ps = [P0.copy() for _ in range(3)]
        Ps[0][3:, 3:] = np.minimum(Ps[0][3:, 3:], np.eye(3) * 0.02)
        mu = np.array([0.34, 0.33, 0.33], dtype=np.float64) if mu0 is None else mu0.astype(np.float64).copy()
        mu /= np.sum(mu)
        return xs, Ps, mu

    def step(
        self,
        xs: list[np.ndarray],
        Ps: list[np.ndarray],
        mu: np.ndarray,
        z: np.ndarray | None = None,
        R_meas: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        # IMM interaction / mixing.
        c_bar = self.Pi.T @ mu
        c_bar = np.maximum(c_bar, 1e-12)
        xs_mix, Ps_mix = [], []
        for j in range(3):
            w_ij = (self.Pi[:, j] * mu) / c_bar[j]
            x_0j = sum(w_ij[i] * xs[i] for i in range(3))
            P_0j = np.zeros((6, 6), dtype=np.float64)
            for i in range(3):
                dx = xs[i] - x_0j
                P_0j += w_ij[i] * (Ps[i] + np.outer(dx, dx))
            xs_mix.append(x_0j)
            Ps_mix.append(P_0j)

        # Mode-conditioned prediction.
        xs_pred, Ps_pred, Fs = [], [], []
        for j in range(3):
            fn = self.step_fns[j]
            F_j = _numerical_jacobian(fn, xs_mix[j])
            x_pj = fn(xs_mix[j])
            P_pj = F_j @ Ps_mix[j] @ F_j.T + self.Q_modes[j]
            xs_pred.append(x_pj)
            Ps_pred.append(0.5 * (P_pj + P_pj.T))
            Fs.append(F_j)

        x_prior = sum(c_bar[j] * xs_pred[j] for j in range(3))
        P_prior = np.zeros((6, 6), dtype=np.float64)
        F_eff = np.zeros((6, 6), dtype=np.float64)
        for j in range(3):
            dx = xs_pred[j] - x_prior
            P_prior += c_bar[j] * (Ps_pred[j] + np.outer(dx, dx))
            F_eff += c_bar[j] * Fs[j]

        if z is None or R_meas is None:
            return {
                "xs": xs_pred,
                "Ps": Ps_pred,
                "mu": c_bar,
                "x_comb": x_prior,
                "P_comb": P_prior,
                "x_prior": x_prior,
                "P_prior": P_prior,
                "F_eff": F_eff,
            }

        # Mode-conditioned measurement update + likelihood scoring.
        xs_upd, Ps_upd = [], []
        log_liks = np.zeros(3, dtype=np.float64)
        I6 = np.eye(6, dtype=np.float64)

        for j in range(3):
            innov = z - (self.H @ xs_pred[j])
            S = self.H @ Ps_pred[j] @ self.H.T + R_meas
            S = 0.5 * (S + S.T)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                S = S + np.eye(3) * 1e-6
                _, logdet = np.linalg.slogdet(S)
            d_m2 = float(innov @ np.linalg.solve(S, innov))
            log_liks[j] = -0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + d_m2)

            K = Ps_pred[j] @ self.H.T @ np.linalg.inv(S)
            x_uj = xs_pred[j] + K @ innov
            IKH = I6 - K @ self.H
            P_uj = IKH @ Ps_pred[j] @ IKH.T + K @ R_meas @ K.T
            xs_upd.append(x_uj)
            Ps_upd.append(0.5 * (P_uj + P_uj.T))

        log_post = log_liks + np.log(c_bar)
        log_post -= np.max(log_post)
        mu_new = np.exp(log_post)
        mu_new = np.maximum(mu_new, 1e-4)
        mu_new /= np.sum(mu_new)

        x_comb = sum(mu_new[j] * xs_upd[j] for j in range(3))
        P_comb = np.zeros((6, 6), dtype=np.float64)
        for j in range(3):
            dx = xs_upd[j] - x_comb
            P_comb += mu_new[j] * (Ps_upd[j] + np.outer(dx, dx))

        return {
            "xs": xs_upd,
            "Ps": Ps_upd,
            "mu": mu_new,
            "x_comb": x_comb,
            "P_comb": P_comb,
            "x_prior": x_prior,
            "P_prior": P_prior,
            "F_eff": F_eff,
        }


def segment_observations(
    P_g: np.ndarray,
    R_meas: np.ndarray,
    valid: np.ndarray,
    fps: float = CONFIG.video.fps,
    max_gap_frames: int = 10,
    border_clipped: np.ndarray | None = None,
) -> list[list[int]]:
    """Partition valid observations at long out-of-frame gaps and persistent kick impulses."""
    obs_idx = np.where(valid)[0].tolist()
    if not obs_idx:
        return []

    gap_runs: list[list[int]] = [[obs_idx[0]]]
    for f in obs_idx[1:]:
        if f - gap_runs[-1][-1] > max_gap_frames:
            gap_runs.append([f])
        else:
            gap_runs[-1].append(f)

    segments: list[list[int]] = []
    dt = 1.0 / fps

    for run in gap_runs:
        if len(run) < 12:
            segments.append(run)
            continue

        cur_seg = [run[0]]
        first_unclip = next((f for f in run if border_clipped is None or not bool(border_clipped[f])), run[0])
        slow_mode = not bool(border_clipped[run[0]]) if border_clipped is not None else True
        anchor_xy = P_g[first_unclip, :2].copy()
        n_slow = 1 if slow_mode else 0

        i = 1
        while i < len(run):
            f = run[i]
            if slow_mode and n_slow >= 10 and i + 2 < len(run):
                f1, f2 = run[i + 1], run[i + 2]
                v0 = (P_g[f, :2] - P_g[cur_seg[-1], :2]) / max((f - cur_seg[-1]) * dt, dt)
                v1 = (P_g[f1, :2] - P_g[f, :2]) / max((f1 - f) * dt, dt)
                v2 = (P_g[f2, :2] - P_g[f1, :2]) / max((f2 - f1) * dt, dt)
                disp = float(np.linalg.norm(P_g[f1, :2] - anchor_xy))
                dir_ok = float(np.dot(v0, v1)) > 0.0 and float(np.dot(v1, v2)) > 0.0
                if (
                    float(np.linalg.norm(v0)) > 2.0
                    and float(np.linalg.norm(v1)) > 2.0
                    and float(np.linalg.norm(v2)) > 2.0
                    and disp > 0.25
                    and dir_ok
                ):
                    segments.append(cur_seg)
                    cur_seg = [f]
                    slow_mode = False
                    i += 1
                    continue

            if slow_mode and (border_clipped is None or not bool(border_clipped[f])):
                sig_step = float(np.sqrt(max(R_meas[f, 0, 0] + R_meas[f, 1, 1], 1e-4)))
                if float(np.linalg.norm(P_g[f, :2] - anchor_xy)) < max(0.15, 3.0 * sig_step):
                    n_slow += 1
                    anchor_xy = anchor_xy + (P_g[f, :2] - anchor_xy) / n_slow
            cur_seg.append(f)
            i += 1

        segments.append(cur_seg)
    return segments


def _estimate_initial_state(
    seg_obs: list[int],
    P_g: np.ndarray,
    dt: float,
    ball: BallConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = seg_obs[0]
    p0 = P_g[f0].copy()
    if len(seg_obs) >= 4:
        f_end = seg_obs[min(4, len(seg_obs) - 1)]
        v0 = (P_g[f_end] - p0) / max((f_end - f0) * dt, dt)
    else:
        v0 = np.zeros(3, dtype=np.float64)

    spd = float(np.linalg.norm(v0[:2]))
    if spd < 0.35:
        v0 = np.zeros(3, dtype=np.float64)
        mu0 = np.array([0.85, 0.12, 0.03], dtype=np.float64)
    elif p0[2] > ball.radius_m + 0.15 and v0[2] > 0.3:
        mu0 = np.array([0.03, 0.12, 0.85], dtype=np.float64)
    else:
        mu0 = np.array([0.05, 0.85, 0.10], dtype=np.float64)

    P0 = np.diag([0.15, 0.15, 0.15, 2.0, 2.0, 2.0])
    return np.concatenate([p0, v0]), P0, mu0


def run_imm_rts_segment(
    seg_obs: list[int],
    P_g: np.ndarray,
    R_meas: np.ndarray,
    dt: float,
    ball: BallConfig = CONFIG.ball,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run forward 3-mode IMM filter and fixed-interval RTS backward smoother over one segment."""
    f_start, f_end = seg_obs[0], seg_obs[-1]
    frames = np.arange(f_start, f_end + 1, dtype=int)
    m = len(frames)
    obs_set = set(seg_obs)

    x0, P0, mu0 = _estimate_initial_state(seg_obs, P_g, dt, ball)
    imm = IMMTracker3D(dt=dt, ball=ball)
    xs, Ps, mu = imm.init_state(x0, P0, mu0)

    x_filt = np.zeros((m, 6), dtype=np.float64)
    P_filt = np.zeros((m, 6, 6), dtype=np.float64)
    x_prior = np.zeros((m, 6), dtype=np.float64)
    P_prior = np.zeros((m, 6, 6), dtype=np.float64)
    F_eff = np.zeros((m, 6, 6), dtype=np.float64)
    mus = np.zeros((m, 3), dtype=np.float64)

    # First frame update directly from initial prior.
    step0 = imm.step(xs, Ps, mu, z=P_g[f_start], R_meas=R_meas[f_start])
    xs, Ps, mu = step0["xs"], step0["Ps"], step0["mu"]
    x_filt[0], P_filt[0] = step0["x_comb"], step0["P_comb"]
    x_prior[0], P_prior[0], F_eff[0] = x0, P0, np.eye(6)
    mus[0] = mu

    for k in range(1, m):
        f = int(frames[k])
        z_k = P_g[f] if f in obs_set else None
        R_k = R_meas[f] if f in obs_set else None
        out = imm.step(xs, Ps, mu, z=z_k, R_meas=R_k)
        xs, Ps, mu = out["xs"], out["Ps"], out["mu"]
        x_filt[k] = out["x_comb"]
        P_filt[k] = out["P_comb"]
        x_prior[k] = out["x_prior"]
        P_prior[k] = out["P_prior"]
        F_eff[k] = out["F_eff"]
        mus[k] = mu

    # Fixed-interval RTS backward pass.
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for k in range(m - 2, -1, -1):
        G = P_filt[k] @ F_eff[k + 1].T @ np.linalg.pinv(P_prior[k + 1])
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - x_prior[k + 1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k + 1] - P_prior[k + 1]) @ G.T

    return frames, x_filt, x_smooth, P_smooth, mus, np.array([f in obs_set for f in frames], dtype=bool)


def _compute_reproj_errors(
    frames: np.ndarray,
    pos_g: np.ndarray,
    s1: Stage1Result,
    s3: Stage3Result,
    camera: CameraConfig = CONFIG.camera,
) -> np.ndarray:
    K = camera.K
    errs = np.full(len(frames), np.nan, dtype=np.float64)
    for i, f in enumerate(frames):
        if not s1.detected[f] or not s3.pose_valid[f]:
            continue
        Pc = s3.R_gc[f].T @ (pos_g[i] - s3.C_g[f])
        if Pc[2] <= 0.1:
            continue
        proj = K @ Pc
        uv_hat = proj[:2] / proj[2]
        errs[i] = float(np.linalg.norm(uv_hat - s1.center_uv[f]))
    return errs


def refine_segment_image_space(
    frames: np.ndarray,
    x_smooth: np.ndarray,
    mus: np.ndarray,
    s1: Stage1Result,
    s3: Stage3Result,
    dt: float,
    camera: CameraConfig = CONFIG.camera,
    ball: BallConfig = CONFIG.ball,
) -> tuple[np.ndarray, bool, float, float]:
    """Refine dynamic rolling segments directly against unclipped 2D pixel rays via soft_l1."""
    imm_errs = _compute_reproj_errors(frames, x_smooth[:, :3], s1, s3, camera)
    unclip_mask = np.array([bool(s1.detected[f] and not s1.border_clipped[f] and s3.pose_valid[f]) for f in frames])
    med_imm = float(np.nanmedian(imm_errs[unclip_mask])) if np.any(unclip_mask) else float("nan")

    med_speed = float(np.median(np.linalg.norm(x_smooth[:, 3:5], axis=1)))
    air_mean = float(np.mean(mus[:, 2]))
    if int(unclip_mask.sum()) < 15 or med_speed < 0.5 or air_mean > 0.4:
        return x_smooth, False, med_imm, med_imm

    K = camera.K
    fit_idx = np.where(unclip_mask)[0]
    m = len(frames)

    def _roll_states(params: np.ndarray) -> np.ndarray:
        px0, py0, vx0, vy0, mu_r, c_lin = params
        traj = np.zeros((m, 6), dtype=np.float64)
        traj[0] = [px0, py0, ball.radius_m, vx0, vy0, 0.0]
        for k in range(1, m):
            traj[k] = step_rolling(traj[k - 1], dt, mu_r=float(mu_r), c_lin=float(c_lin), ball=ball)
        return traj

    def _residuals(params: np.ndarray) -> np.ndarray:
        traj = _roll_states(params)
        res = []
        for idx in fit_idx:
            f = frames[idx]
            Pc = s3.R_gc[f].T @ (traj[idx, :3] - s3.C_g[f])
            proj = K @ Pc
            uv_hat = proj[:2] / max(proj[2], 0.1)
            res.extend((uv_hat - s1.center_uv[f]).tolist())
        return np.asarray(res, dtype=np.float64)

    x0 = np.array([x_smooth[0, 0], x_smooth[0, 1], x_smooth[0, 3], x_smooth[0, 4], ball.rolling_friction, 0.0])
    lb = np.array([x0[0] - 1.5, x0[1] - 1.5, -20.0, -20.0, 0.005, 0.0])
    ub = np.array([x0[0] + 1.5, x0[1] + 1.5, 20.0, 20.0, 0.15, 0.5])

    opt = least_squares(_residuals, x0=x0, bounds=(lb, ub), loss="soft_l1", f_scale=2.0, max_nfev=120)
    traj_opt = _roll_states(opt.x)
    opt_errs = _compute_reproj_errors(frames, traj_opt[:, :3], s1, s3, camera)
    med_opt = float(np.nanmedian(opt_errs[unclip_mask])) if np.any(unclip_mask) else float("inf")

    if np.isfinite(med_opt) and med_opt < 5.0 and med_opt <= med_imm + 0.5:
        return traj_opt, True, med_imm, med_opt
    return x_smooth, False, med_imm, med_opt


def predict_forward_rollout(
    x_filt: np.ndarray,
    mu: np.ndarray,
    dt: float,
    steps: int = 24,
    ball: BallConfig = CONFIG.ball,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll state forward using active IMM mode, tracking airborne landing point if applicable."""
    mode = int(np.argmax(mu))
    traj = np.zeros((steps, 3), dtype=np.float64)
    landing = np.full(3, np.nan, dtype=np.float64)
    state = x_filt.copy()

    for s in range(steps):
        if mode == 0:
            state = step_stationary(state, dt, ball)
        elif mode == 1:
            state = step_rolling(state, dt, ball.rolling_friction, 0.0, ball)
        else:
            prev_pos = state[:3].copy()
            state = step_airborne(state, dt, None, ball)
            if state[2] <= ball.radius_m and state[5] < 0.0:
                if np.isnan(landing[0]):
                    frac = (prev_pos[2] - ball.radius_m) / max(prev_pos[2] - state[2], 1e-6)
                    landing = prev_pos + frac * (state[:3] - prev_pos)
                    landing[2] = ball.radius_m
                state[2] = ball.radius_m
                state[5] = -0.70 * state[5]
                state[3:5] *= 0.85
        traj[s] = state[:3]
    return traj, landing


def save_stage4_diagnostics(
    df_tracked: pd.DataFrame,
    df_fused: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Plot smoothed trajectory/velocities, IMM mode probabilities, and pixel reprojection error."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames = df_tracked["frame_idx"].to_numpy()
    tracked_mask = df_tracked["track_status"].to_numpy() != "gap"
    g0, g1 = CONFIG.video.gap_frames

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), sharex=True)
    for ax in axes:
        ax.axvspan(g0, g1, color="gray", alpha=0.12)

    axes[0].plot(frames, df_fused["X_g_sph"], ".", color="tab:blue", ms=3, alpha=0.35, label="X_g raw sphere")
    axes[0].plot(frames, df_tracked["X_g"], color="tab:blue", lw=1.8, label="X_g smoothed")
    axes[0].plot(frames, df_tracked["Y_g"], color="tab:green", lw=1.8, label="Y_g smoothed")
    axes[0].plot(frames, df_tracked["Z_g"], color="tab:orange", lw=1.5, label="Z_g smoothed")
    axes[0].axhline(CONFIG.ball.radius_m, color="black", ls=":", lw=1.0)
    axes[0].set_ylabel("Ground Position (m)")
    axes[0].set_title("Stage 4: Ground-Frame Smoothed Trajectory, Speed & IMM Mode Probabilities")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(frames, df_tracked["speed_g"], color="tab:red", lw=1.8, label="World speed |v_g| (m/s)")
    axes[1].plot(frames, df_tracked["speed_c_rel"], "--", color="tab:purple", lw=1.3, alpha=0.85, label="Cam-relative speed (m/s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(alpha=0.3)

    axes[2].plot(frames, df_tracked["p_stationary"], color="tab:green", lw=1.5, label="P(stationary)")
    axes[2].plot(frames, df_tracked["p_rolling"], color="tab:orange", lw=1.5, label="P(rolling)")
    axes[2].plot(frames, df_tracked["p_airborne"], color="tab:blue", lw=1.2, label="P(airborne)")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_ylabel("IMM Probability")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="center right", fontsize=8.5)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_trajectory_and_velocity.png", dpi=150)
    plt.close(fig)

    errs = df_tracked["reproj_err_px"].to_numpy()
    valid_e = errs[np.isfinite(errs)]
    med_e = float(np.median(valid_e)) if len(valid_e) else float("nan")
    p95_e = float(np.percentile(valid_e, 95)) if len(valid_e) else float("nan")

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.8, 1.0]})
    axes2[0].axvspan(g0, g1, color="gray", alpha=0.12)
    axes2[0].plot(frames, errs, color="tab:blue", lw=1.3)
    axes2[0].axhline(med_e, color="tab:green", ls="--", lw=1.2, label=f"Median: {med_e:.2f} px")
    axes2[0].axhline(p95_e, color="tab:orange", ls=":", lw=1.2, label=f"P95: {p95_e:.2f} px")
    axes2[0].set_xlabel("Frame Index")
    axes2[0].set_ylabel("Reprojection Error (px)")
    axes2[0].set_title("2D Reprojection Consistency Over Time")
    axes2[0].legend(loc="upper right")
    axes2[0].grid(alpha=0.3)

    axes2[1].hist(valid_e, bins=25, color="tab:blue", alpha=0.8, edgecolor="white")
    axes2[1].axvline(med_e, color="tab:green", ls="--", lw=1.5)
    axes2[1].set_xlabel("Pixel Error (px)")
    axes2[1].set_ylabel("Frames")
    axes2[1].set_title("Error Distribution")
    axes2[1].grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(plots_dir / "stage4_reprojection_error.png", dpi=150)
    plt.close(fig2)


def run_stage4(profile_name: str = "local", pred_steps: int = 24) -> tuple[pd.DataFrame, dict]:
    """Run Ray-Ground fusion, segmented IMM+RTS tracking, image-space refinement & forward rollout."""
    CONFIG.paths.ensure_dirs(profile_name)
    cache_dir = CONFIG.paths.cache_dir(profile_name)
    s1 = Stage1Result.load(cache_dir)
    df2 = pd.read_csv(cache_dir / "stage2_3d.csv")
    s3 = Stage3Result.load(cache_dir)

    df_fused, R_meas_g = fuse_sphere_with_ground_plane(s1, df2, s3)
    df_fused.to_csv(cache_dir / "stage2_3d_fused.csv", index=False)

    n = len(df_fused)
    dt = 1.0 / CONFIG.video.fps
    P_g = df_fused[["X_g_fused", "Y_g_fused", "Z_g_fused"]].to_numpy(dtype=np.float64)
    valid_obs = np.isfinite(P_g[:, 0]) & np.isfinite(R_meas_g[:, 0, 0])

    segments = segment_observations(P_g, R_meas_g, valid_obs, fps=CONFIG.video.fps, border_clipped=s1.border_clipped)

    seg_id_arr = np.full(n, -1, dtype=np.int32)
    status_arr = np.full(n, "gap", dtype="U16")
    X_g_arr = np.full((n, 3), np.nan, dtype=np.float64)
    V_g_arr = np.full((n, 3), np.nan, dtype=np.float64)
    P_c_arr = np.full((n, 3), np.nan, dtype=np.float64)
    V_c_arr = np.full((n, 3), np.nan, dtype=np.float64)
    mu_arr = np.full((n, 3), np.nan, dtype=np.float64)
    reproj_arr = np.full(n, np.nan, dtype=np.float64)

    pred_traj_g = np.full((n, pred_steps, 3), np.nan, dtype=np.float64)
    pred_uv = np.full((n, pred_steps, 2), np.nan, dtype=np.float64)
    pred_valid = np.zeros(n, dtype=bool)
    landing_g = np.full((n, 3), np.nan, dtype=np.float64)

    K = CONFIG.camera.K
    for sid, seg_obs in enumerate(segments):
        frames_seg, x_filt, x_smooth, _, mus, obs_flags = run_imm_rts_segment(
            seg_obs, P_g, R_meas_g, dt, CONFIG.ball
        )
        x_final, opt_used, err_imm, err_opt = refine_segment_image_space(
            frames_seg, x_smooth, mus, s1, s3, dt, CONFIG.camera, CONFIG.ball
        )
        print(
            f"Segment {sid}: F{frames_seg[0]}..F{frames_seg[-1]} ({len(frames_seg)}f) | "
            f"IMM_err={err_imm:.2f}px, Opt_err={err_opt:.2f}px -> {'ImageOpt' if opt_used else 'IMM-RTS'}"
        )

        # Back-project smoothed world positions to camera coordinates and compute relative velocities.
        pc_seg = np.array(
            [s3.R_gc[f].T @ (x_final[i, :3] - s3.C_g[f]) for i, f in enumerate(frames_seg)],
            dtype=np.float64,
        )
        vc_seg = np.gradient(pc_seg, dt, axis=0) if len(frames_seg) >= 2 else np.zeros_like(pc_seg)
        errs_seg = _compute_reproj_errors(frames_seg, x_final[:, :3], s1, s3, CONFIG.camera)

        for i, f in enumerate(frames_seg):
            seg_id_arr[f] = sid
            status_arr[f] = "observed" if obs_flags[i] else "interpolated"
            X_g_arr[f] = x_final[i, :3]
            V_g_arr[f] = x_final[i, 3:6]
            P_c_arr[f] = pc_seg[i]
            V_c_arr[f] = vc_seg[i]
            mu_arr[f] = mus[i]
            reproj_arr[f] = errs_seg[i]

            # Forward rollout from causal filtered state (or refined state when accepted).
            roll_state = x_final[i] if opt_used else x_filt[i]
            traj_f, land_f = predict_forward_rollout(roll_state, mus[i], dt, pred_steps, CONFIG.ball)
            pred_traj_g[f] = traj_f
            landing_g[f] = land_f
            pred_valid[f] = True
            for step_j in range(pred_steps):
                pc_j = s3.R_gc[f].T @ (traj_f[step_j] - s3.C_g[f])
                if pc_j[2] > 0.2:
                    pr = K @ pc_j
                    pred_uv[f, step_j] = pr[:2] / pr[2]

    dist_smooth = np.linalg.norm(P_c_arr, axis=1)
    speed_g = np.linalg.norm(V_g_arr, axis=1)
    speed_c = np.linalg.norm(V_c_arr, axis=1)

    df_tracked = pd.DataFrame(
        {
            "frame_idx": df_fused["frame_idx"].to_numpy(dtype=int),
            "timestamp_s": df_fused["timestamp_s"].to_numpy(dtype=float),
            "segment_id": seg_id_arr,
            "track_status": status_arr,
            "X_g": X_g_arr[:, 0],
            "Y_g": X_g_arr[:, 1],
            "Z_g": X_g_arr[:, 2],
            "Vx_g": V_g_arr[:, 0],
            "Vy_g": V_g_arr[:, 1],
            "Vz_g": V_g_arr[:, 2],
            "speed_g": speed_g,
            "X_c_smooth": P_c_arr[:, 0],
            "Y_c_smooth": P_c_arr[:, 1],
            "Z_c_smooth": P_c_arr[:, 2],
            "distance_smooth_m": dist_smooth,
            "Vx_c_rel": V_c_arr[:, 0],
            "Vy_c_rel": V_c_arr[:, 1],
            "Vz_c_rel": V_c_arr[:, 2],
            "speed_c_rel": speed_c,
            "p_stationary": mu_arr[:, 0],
            "p_rolling": mu_arr[:, 1],
            "p_airborne": mu_arr[:, 2],
            "C_x_g": s3.C_g[:, 0],
            "C_y_g": s3.C_g[:, 1],
            "C_z_g": s3.C_g[:, 2],
            "reproj_err_px": reproj_arr,
        }
    )
    df_tracked.to_csv(cache_dir / "stage4_tracked.csv", index=False)

    preds = {
        "pred_dt_s": np.float64(dt),
        "pred_steps": np.int32(pred_steps),
        "pred_traj_g": pred_traj_g,
        "pred_uv": pred_uv,
        "pred_valid": pred_valid,
        "landing_point_g": landing_g,
    }
    np.savez_compressed(cache_dir / "stage4_predictions.npz", **preds)
    save_stage4_diagnostics(df_tracked, df_fused, CONFIG.paths.plots_dir)
    return df_tracked, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: Ground fusion, 3-mode IMM+RTS & trajectory rollout.")
    parser.add_argument("--profile", default="local", choices=("local", "gpu"))
    parser.add_argument("--pred-steps", type=int, default=24)
    args = parser.parse_args()

    df, _ = run_stage4(args.profile, pred_steps=args.pred_steps)
    s1 = Stage1Result.load(CONFIG.paths.cache_dir(args.profile))
    s0, s1_end = CONFIG.video.stationary_frames
    stat = df.iloc[s0 : s1_end + 1]
    stat_ok = stat[stat["track_status"] != "gap"]
    stat_std_cm = stat_ok[["X_g", "Y_g"]].to_numpy().std(axis=0) * 100.0

    errs_all = df["reproj_err_px"].to_numpy(dtype=np.float64)
    valid_all = errs_all[np.isfinite(errs_all)]
    valid_unclip = errs_all[np.isfinite(errs_all) & (~s1.border_clipped)]
    print(f"Pre-kick smoothed XY std ({s0}-{s1_end}): {stat_std_cm.round(2)} cm")
    print(
        f"Reprojection error (unclipped): median={np.median(valid_unclip):.2f} px, "
        f"p95={np.percentile(valid_unclip, 95):.2f} px, max={np.max(valid_unclip):.2f} px"
    )
    print(
        f"Reprojection error (all):       median={np.median(valid_all):.2f} px, "
        f"p95={np.percentile(valid_all, 95):.2f} px, max={np.max(valid_all):.2f} px"
    )
    for sid in sorted(s for s in df["segment_id"].unique() if s >= 0):
        sub = df[df["segment_id"] == sid]
        print(
            f"  Seg {sid} (F{int(sub['frame_idx'].min())}..F{int(sub['frame_idx'].max())}): "
            f"speed med={sub['speed_g'].median():.2f}, max={sub['speed_g'].max():.2f} m/s | "
            f"P_stat={sub['p_stationary'].mean():.2f}, P_roll={sub['p_rolling'].mean():.2f}, P_air={sub['p_airborne'].mean():.2f}"
        )


if __name__ == "__main__":
    main()
