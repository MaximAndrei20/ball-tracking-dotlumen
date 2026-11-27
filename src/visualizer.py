# Stage 5 outputs. Renders the trajectory overlay, 2D top-view map, side-by-side video, and exports Excel and CSV telemetry.

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.config import CONFIG
from src.detector_2d import Stage1Result
from src.ego_motion import Stage3Result


def _load_stage_data(
    profile_name: str,
) -> tuple[Stage1Result, pd.DataFrame, pd.DataFrame, Stage3Result, pd.DataFrame, dict[str, np.ndarray]]:
    cache_dir = CONFIG.paths.cache_dir(profile_name)
    s1 = Stage1Result.load(cache_dir)
    df2 = pd.read_csv(cache_dir / "stage2_3d.csv")
    df2f = pd.read_csv(cache_dir / "stage2_3d_fused.csv")
    s3 = Stage3Result.load(cache_dir)
    df4 = pd.read_csv(cache_dir / "stage4_tracked.csv")
    with np.load(cache_dir / "stage4_predictions.npz", allow_pickle=False) as d:
        preds = {k: d[k] for k in d.files}
    return s1, df2, df2f, s3, df4, preds


def _open_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {path}")
    return writer


def _project_points_to_image(
    pts_g: np.ndarray,
    R_gc: np.ndarray,
    C_g: np.ndarray,
    K: np.ndarray,
    min_depth_m: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) ground points into camera frame K * R_gc^T * (P_g - C_g)."""
    if len(pts_g) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)
    pc = (R_gc.T @ (pts_g - C_g).T).T
    valid = np.isfinite(pc[:, 2]) & (pc[:, 2] > min_depth_m)
    uv = np.full((len(pts_g), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        proj = (K @ pc[valid].T).T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    return uv, valid


def _draw_dashed_polyline(
    img: np.ndarray,
    pts_px: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 7,
) -> None:
    """Draw a dashed polyline while filtering NaNs and extreme projections."""
    pts = [p for p in pts_px if np.all(np.isfinite(p)) and -4000 < p[0] < 6000 and -4000 < p[1] < 6000]
    if len(pts) < 2:
        return
    draw_on = True
    rem = float(dash_len)
    for i in range(len(pts) - 1):
        p0 = np.asarray(pts[i], dtype=np.float64)
        p1 = np.asarray(pts[i + 1], dtype=np.float64)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-3:
            continue
        dir_v = seg / seg_len
        pos = 0.0
        while pos < seg_len:
            step = min(rem, seg_len - pos)
            if draw_on and step >= 1.0:
                a = np.round(p0 + pos * dir_v).astype(int)
                b = np.round(p0 + (pos + step) * dir_v).astype(int)
                cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
            pos += step
            rem -= step
            if rem <= 1e-4:
                draw_on = not draw_on
                rem = float(dash_len if draw_on else gap_len)


def _speed_to_bgr(speed_mps: float) -> tuple[int, int, int]:
    """Map speed in m/s to BGR ramp: teal (0) -> lime/yellow (1.8) -> orange/red (3.8+)."""
    if not np.isfinite(speed_mps):
        return (180, 180, 180)
    t = float(np.clip(speed_mps / 3.8, 0.0, 1.0))
    if t < 0.5:
        u = t * 2.0
        b = int(round(230 * (1.0 - 0.7 * u)))
        g = int(round(190 + 60 * u))
        r = int(round(50 + 200 * u))
    else:
        u = (t - 0.5) * 2.0
        b = int(round(69 * (1.0 - u)))
        g = int(round(250 * (1.0 - 0.72 * u)))
        r = 255
    return (b, g, r)


def _mode_label_and_color(row: pd.Series) -> tuple[str, float, tuple[int, int, int]]:
    if str(row["track_status"]) == "gap":
        return "OUT OF FRAME", 0.0, (140, 140, 155)
    probs = [
        ("STATIONARY", float(row["p_stationary"]), (110, 220, 120)),
        ("ROLLING", float(row["p_rolling"]), (60, 185, 255)),
        ("AIRBORNE", float(row["p_airborne"]), (255, 165, 65)),
    ]
    return max(probs, key=lambda item: item[1])


class TopViewMapRenderer:
    """Camera-independent 2D metric floor map renderer with SfM terrace points & FOV frustum."""

    def __init__(
        self,
        s3: Stage3Result,
        df4: pd.DataFrame,
        preds: dict[str, np.ndarray],
        width: int = 720,
        height: int = 720,
    ) -> None:
        self.s3 = s3
        self.df4 = df4
        self.preds = preds
        self.width = width
        self.height = height
        self.header_h = 56
        self.footer_h = 46

        obs = df4[df4["track_status"] != "gap"]
        cam_xy = s3.C_g[:, :2]
        x_pts = np.concatenate([cam_xy[:, 0], obs["X_g"].to_numpy()])
        self.x_min = min(float(np.min(x_pts)), 0.0) - 0.85
        self.x_max = min(max(float(np.max(x_pts)) + 2.4, 9.6), 12.0)

        avail_h = self.height - self.header_h - self.footer_h - 24
        self.scale_px_m = avail_h / max(self.x_max - self.x_min, 2.0)
        self.y_center = 0.55

        # Compute stationary-ball drift readout dynamically from active stationary IMM predictions.
        stat_ok = df4[(df4["track_status"] != "gap") & (df4["p_stationary"] > 0.7)]
        self.stat_span: tuple[int, int] | None = (
            (int(stat_ok["frame_idx"].min()), int(stat_ok["frame_idx"].max()))
            if len(stat_ok) >= 2
            else None
        )
        self.stat_std_cm = (
            stat_ok[["X_g", "Y_g"]].to_numpy().std(axis=0) * 100.0
            if len(stat_ok) >= 2
            else None
        )

        self._static_canvas = self._build_static_canvas()
        self._cam_trail_px = self._world_to_px(self.s3.C_g[:, :2])
        self._segments_cache = self._build_segment_cache()

    def _world_to_px(self, xy_g: np.ndarray) -> np.ndarray:
        """Map (N, 2) world XY to canvas pixels: +X_g UP, +Y_g LEFT, -Y_g RIGHT."""
        arr = np.atleast_2d(np.asarray(xy_g, dtype=np.float64))
        px = (self.width * 0.5) - (arr[:, 1] - self.y_center) * self.scale_px_m
        py = (self.height - self.footer_h - 12) - (arr[:, 0] - self.x_min) * self.scale_px_m
        out = np.column_stack([px, py])
        return out[0] if np.asarray(xy_g).ndim == 1 else out

    def _build_segment_cache(self) -> list[dict]:
        cache = []
        sids = sorted(int(s) for s in self.df4["segment_id"].unique() if int(s) >= 0)
        for sid in sids:
            sub = self.df4[self.df4["segment_id"] == sid].sort_values("frame_idx")
            frames = sub["frame_idx"].to_numpy(dtype=int)
            xy_px = self._world_to_px(sub[["X_g", "Y_g"]].to_numpy(dtype=np.float64))
            speeds = sub["speed_g"].to_numpy(dtype=np.float64)
            cache.append({"sid": sid, "frames": frames, "xy_px": xy_px, "speeds": speeds})
        return cache

    def _build_static_canvas(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), (32, 24, 20), dtype=np.uint8)
        top_y = self.header_h
        bot_y = self.height - self.footer_h

        # 1-meter metric grid inside map area.
        y_half_span = (self.width * 0.5) / self.scale_px_m
        y_lo = math.ceil(self.y_center - y_half_span)
        y_hi = math.floor(self.y_center + y_half_span)
        x_lo = math.ceil(self.x_min)
        x_hi = math.floor(self.x_max)

        for xg in range(x_lo, x_hi + 1):
            p0 = self._world_to_px(np.array([float(xg), y_lo - 1.0]))
            p1 = self._world_to_px(np.array([float(xg), y_hi + 1.0]))
            py = int(round(p0[1]))
            if top_y < py < bot_y:
                col = (52, 42, 36) if xg == 0 else (40, 32, 26)
                cv2.line(canvas, (0, py), (self.width, py), col, 1)
                cv2.putText(canvas, f"{xg}m", (8, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 140, 155), 1, cv2.LINE_AA)

        for yg in range(y_lo, y_hi + 1):
            p0 = self._world_to_px(np.array([self.x_min, float(yg)]))
            px = int(round(p0[0]))
            if 0 < px < self.width:
                col = (52, 42, 36) if yg == 0 else (40, 32, 26)
                cv2.line(canvas, (px, top_y), (px, bot_y), col, 1)
                cv2.putText(canvas, f"{yg:+}m" if yg != 0 else "0m", (px + 3, bot_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 140, 155), 1, cv2.LINE_AA)

        # Filtered static terrace SfM point cloud backdrop.
        pts = self.s3.sparse_points_g
        if len(pts) > 0:
            keep = (pts[:, 2] >= -0.25) & (pts[:, 2] <= 2.2)
            pts_f = pts[keep]
            uv_pts = self._world_to_px(pts_f[:, :2])
            for idx, (pt_px, z_val) in enumerate(zip(uv_pts, pts_f[:, 2])):
                ix, iy = int(round(pt_px[0])), int(round(pt_px[1]))
                if 2 <= ix < self.width - 2 and (top_y + 2) <= iy < (bot_y - 2):
                    col = (155, 138, 118) if z_val >= 0.25 else (92, 82, 70)
                    r = 2 if z_val >= 0.25 else 1
                    cv2.circle(canvas, (ix, iy), r, col, -1, cv2.LINE_AA)

        # Header & pre-kick camera-decoupling verification badge.
        cv2.rectangle(canvas, (0, 0), (self.width, self.header_h), (24, 18, 14), -1)
        cv2.line(canvas, (0, self.header_h), (self.width, self.header_h), (70, 62, 52), 1)
        cv2.putText(
            canvas,
            "2D TOP-VIEW MAP (GROUND FRAME)",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 244, 248),
            2,
            cv2.LINE_AA,
        )
        drift_text = (
            f"Stationary drift (F{self.stat_span[0]}-{self.stat_span[1]}): "
            f"X={self.stat_std_cm[0]:.2f}cm  Y={self.stat_std_cm[1]:.2f}cm"
            if self.stat_span is not None and self.stat_std_cm is not None
            else "Camera-independent ground frame"
        )
        cv2.putText(
            canvas,
            drift_text,
            (14, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (110, 230, 150),
            1,
            cv2.LINE_AA,
        )

        # Compass & 1 m scale bar in top-right of map area.
        cx_c, cy_c = self.width - 44, top_y + 42
        cv2.arrowedLine(canvas, (cx_c, cy_c), (cx_c, cy_c - 26), (130, 220, 255), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(canvas, (cx_c, cy_c), (cx_c - 26, cy_c), (130, 255, 170), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(canvas, "+X", (cx_c - 9, cy_c - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "+Y", (cx_c - 48, cy_c + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 255, 170), 1, cv2.LINE_AA)

        bar_len = int(round(self.scale_px_m))
        bx1, by = self.width - 20, top_y + 68
        bx0 = bx1 - bar_len
        cv2.line(canvas, (bx0, by), (bx1, by), (220, 225, 232), 2)
        cv2.line(canvas, (bx0, by - 4), (bx0, by + 4), (220, 225, 232), 2)
        cv2.line(canvas, (bx1, by - 4), (bx1, by + 4), (220, 225, 232), 2)
        cv2.putText(canvas, "1.0 m", (bx0 + bar_len // 2 - 18, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 225, 232), 1, cv2.LINE_AA)

        # Compact legend in top-left under header.
        lx, ly = 12, top_y + 10
        cv2.rectangle(canvas, (lx, ly), (lx + 178, ly + 82), (24, 18, 14), -1)
        cv2.rectangle(canvas, (lx, ly), (lx + 178, ly + 82), (65, 58, 48), 1)
        items = [
            ((110, 210, 110), "Camera FOV (69.4 deg)"),
            ((245, 200, 80), "Camera Trajectory"),
            ((220, 90, 255), "Ball Position"),
            ((0, 235, 255), "0.8s Prediction"),
        ]
        for idx, (col, text) in enumerate(items):
            yy = ly + 17 + idx * 18
            cv2.circle(canvas, (lx + 12, yy - 4), 4, col, -1, cv2.LINE_AA)
            cv2.putText(canvas, text, (lx + 24, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 230, 238), 1, cv2.LINE_AA)

        return canvas

    def render_frame(self, f: int) -> np.ndarray:
        """Render the ground-aligned top-view map at frame index f."""
        img = self._static_canvas.copy()
        bot_y = self.height - self.footer_h

        # Horizontal FOV wedge projected from exact 3D camera frustum side rays.
        if bool(self.s3.pose_valid[f]):
            C_g = self.s3.C_g[f]
            R_gc = self.s3.R_gc[f]
            half_tan = math.tan(math.radians(CONFIG.camera.hfov_deg * 0.5))
            frustum_pts_g = [C_g[:2]]
            ray_len_m = 8.4
            for tx in np.linspace(-half_tan, half_tan, 15):
                d_g = R_gc @ np.array([tx, 0.0, 1.0], dtype=np.float64)
                d_xy = d_g[:2]
                norm_xy = float(np.linalg.norm(d_xy))
                if norm_xy > 1e-6:
                    frustum_pts_g.append(C_g[:2] + ray_len_m * (d_xy / norm_xy))
            poly_px = np.round(self._world_to_px(np.asarray(frustum_pts_g))).astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_px], (70, 165, 95))
            cv2.addWeighted(overlay, 0.16, img, 0.84, 0.0, img)
            cv2.line(img, tuple(poly_px[0]), tuple(poly_px[1]), (90, 200, 120), 1, cv2.LINE_AA)
            cv2.line(img, tuple(poly_px[0]), tuple(poly_px[-1]), (90, 200, 120), 1, cv2.LINE_AA)

        # Camera path & oriented camera marker.
        valid_cam = np.where(self.s3.pose_valid[: f + 1])[0]
        if len(valid_cam) >= 2:
            pts_c = np.round(self._cam_trail_px[valid_cam]).astype(np.int32)
            cv2.polylines(img, [pts_c], False, (245, 200, 80), 2, cv2.LINE_AA)
        if bool(self.s3.pose_valid[f]):
            c_px = self._cam_trail_px[f]
            fwd_xy = self.s3.R_gc[f, :2, 2]
            fwd_px = np.array([-fwd_xy[1], -fwd_xy[0]], dtype=np.float64)
            fwd_px /= max(float(np.linalg.norm(fwd_px)), 1e-6)
            right_px = np.array([-fwd_px[1], fwd_px[0]], dtype=np.float64)
            tri = np.round(
                np.vstack([c_px + 11 * fwd_px, c_px - 7 * fwd_px + 7 * right_px, c_px - 7 * fwd_px - 7 * right_px])
            ).astype(np.int32)
            cv2.fillConvexPoly(img, tri, (255, 215, 95), cv2.LINE_AA)
            cv2.polylines(img, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Ball segment trajectories up to frame f.
        for seg in self._segments_cache:
            mask = seg["frames"] <= f
            count = int(np.sum(mask))
            if count < 2:
                continue
            pts = seg["xy_px"][mask]
            spds = seg["speeds"][mask]
            is_cur = f <= int(seg["frames"][-1])
            thick = 3 if is_cur else 2
            for j in range(1, count):
                col = _speed_to_bgr(float(spds[j]))
                if not is_cur:
                    col = tuple(int(0.62 * c) for c in col)
                p0 = tuple(np.round(pts[j - 1]).astype(int))
                p1 = tuple(np.round(pts[j]).astype(int))
                cv2.line(img, p0, p1, col, thick, cv2.LINE_AA)

        # Forward predicted rollout & current ball marker.
        row = self.df4.iloc[f]
        if str(row["track_status"]) != "gap":
            if bool(self.preds["pred_valid"][f]):
                pred_xy = self.preds["pred_traj_g"][f, :, :2]
                pred_px = self._world_to_px(pred_xy)
                _draw_dashed_polyline(img, pred_px, (0, 235, 255), thickness=2, dash_len=8, gap_len=5)

            b_px = tuple(np.round(self._world_to_px(np.array([row["X_g"], row["Y_g"]], dtype=np.float64))).astype(int))
            cv2.circle(img, b_px, 11, (220, 90, 255), 2, cv2.LINE_AA)
            cv2.circle(img, b_px, 6, (255, 130, 255), -1, cv2.LINE_AA)
            cv2.circle(img, b_px, 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Bottom status strip.
        cv2.rectangle(img, (0, bot_y), (self.width, self.height), (24, 18, 14), -1)
        cv2.line(img, (0, bot_y), (self.width, bot_y), (70, 62, 52), 1)
        if str(row["track_status"]) == "gap":
            info = f"F{f:03d} ({row['timestamp_s']:.2f}s) | Cam=({row['C_x_g']:.2f}, {row['C_y_g']:.2f})m | BALL OUT OF FRAME"
            tcol = (165, 175, 190)
        else:
            info = (
                f"F{f:03d} ({row['timestamp_s']:.2f}s) | Cam=({row['C_x_g']:.2f},{row['C_y_g']:.2f})m | "
                f"Ball=({row['X_g']:.2f},{row['Y_g']:.2f})m | |v_g|={row['speed_g']:.2f}m/s"
            )
            tcol = (235, 240, 245)
        cv2.putText(img, info, (14, bot_y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.46, tcol, 1, cv2.LINE_AA)
        return img


def _draw_rgb_trajectory_frame(
    frame_bgr: np.ndarray,
    f: int,
    s1: Stage1Result,
    s3: Stage3Result,
    df4: pd.DataFrame,
    preds: dict[str, np.ndarray],
    K: np.ndarray,
) -> np.ndarray:
    """Render trajectory trail, 0.8s dashed prediction, and telemetry HUD on RGB frame."""
    out = frame_bgr.copy()
    row = df4.iloc[f]
    sid = int(row["segment_id"])
    mode_lbl, mode_prob, mode_col = _mode_label_and_color(row)

    # Current segment smoothed ground trail projected into current camera view.
    if sid >= 0 and str(row["track_status"]) != "gap" and bool(s3.pose_valid[f]):
        seg_sub = df4[(df4["segment_id"] == sid) & (df4["frame_idx"] <= f)].sort_values("frame_idx")
        pts_g = seg_sub[["X_g", "Y_g", "Z_g"]].to_numpy(dtype=np.float64)
        spds = seg_sub["speed_g"].to_numpy(dtype=np.float64)
        uv_trail, valid_trail = _project_points_to_image(pts_g, s3.R_gc[f], s3.C_g[f], K)
        n_pts = len(uv_trail)
        for j in range(1, n_pts):
            if valid_trail[j - 1] and valid_trail[j]:
                age = j / max(n_pts - 1, 1)
                thick = 2 if age < 0.5 else 3 if age < 0.85 else 4
                p0 = tuple(np.round(uv_trail[j - 1]).astype(int))
                p1 = tuple(np.round(uv_trail[j]).astype(int))
                cv2.line(out, p0, p1, _speed_to_bgr(float(spds[j])), thick, cv2.LINE_AA)

        # Forward dashed prediction trajectory.
        if bool(preds["pred_valid"][f]):
            _draw_dashed_polyline(out, preds["pred_uv"][f], (0, 235, 255), thickness=2, dash_len=10, gap_len=7)

        # Reprojected 3D smoothed ball sphere vs. raw Stage 1 ellipse.
        if bool(s1.detected[f]):
            cu, cv_ = np.round(s1.center_uv[f]).astype(int)
            axes = np.round(s1.semi_axes_ab[f]).astype(int)
            if np.all(axes > 1):
                cv2.ellipse(out, (cu, cv_), tuple(axes), float(s1.ellipse_angle_deg[f]), 0, 360, (255, 210, 80), 1, cv2.LINE_AA)

        uv_ball, ok_ball = _project_points_to_image(
            np.array([[row["X_g"], row["Y_g"], row["Z_g"]]], dtype=np.float64),
            s3.R_gc[f],
            s3.C_g[f],
            K,
        )
        if ok_ball[0] and np.all(np.isfinite(uv_ball[0])):
            bx, by = int(round(uv_ball[0, 0])), int(round(uv_ball[0, 1]))
            r_px = int(round(CONFIG.camera.f_mean * CONFIG.ball.radius_m / max(float(row["Z_c_smooth"]), 0.3)))
            r_px = int(np.clip(r_px, 6, 90))
            cv2.circle(out, (bx, by), r_px, (110, 245, 130), 2, cv2.LINE_AA)
            cv2.drawMarker(out, (bx, by), (255, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

    # Translucent HUD panel (top-left).
    hud = out.copy()
    cv2.rectangle(hud, (14, 14), (448, 138), (22, 18, 16), -1)
    cv2.addWeighted(hud, 0.72, out, 0.28, 0.0, out)
    cv2.rectangle(out, (14, 14), (448, 138), (85, 80, 72), 1)

    cv2.putText(out, f"F{f:03d}/{len(df4)-1:03d}  t={row['timestamp_s']:.2f}s", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 248, 252), 2, cv2.LINE_AA)
    pill_txt = f"{mode_lbl}" + (f" ({mode_prob:.2f})" if mode_prob > 0.0 else "")
    cv2.putText(out, pill_txt, (252, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_col, 2, cv2.LINE_AA)

    if str(row["track_status"]) == "gap":
        cv2.putText(out, "Cam 3D: --  (out of field of view)", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 190, 200), 1, cv2.LINE_AA)
        cv2.putText(out, "World |v_g|: --   |   Cam |v_c|: --", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 190, 200), 1, cv2.LINE_AA)
        cv2.putText(out, "3D trail & 0.8s rollout pause safely at gap", (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 160, 175), 1, cv2.LINE_AA)
    else:
        cv2.putText(
            out,
            f"Cam XYZ: ({row['X_c_smooth']:+.2f}, {row['Y_c_smooth']:+.2f}, {row['Z_c_smooth']:.2f})m | d={row['distance_smooth_m']:.2f}m",
            (24, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (230, 238, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Speed |v_g|: {row['speed_g']:.2f} m/s  |  |v_c|: {row['speed_c_rel']:.2f} m/s",
            (24, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (130, 240, 185),
            1,
            cv2.LINE_AA,
        )
        err_str = f"{row['reproj_err_px']:.2f} px" if np.isfinite(row["reproj_err_px"]) else "--"
        cv2.putText(
            out,
            f"World: ({row['X_g']:.2f}, {row['Y_g']:.2f}, {row['Z_g']:.2f})m | reproj: {err_str}",
            (24, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (195, 205, 220),
            1,
            cv2.LINE_AA,
        )
    return out


def _save_snapshot_figure(
    snap_frames: dict[int, tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
) -> Path:
    """Save synchronized RGB trajectory + 2D top-view snapshots for documentation."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(snap_frames.items())
    fig, axes = plt.subplots(len(ordered), 2, figsize=(13.5, 4.4 * len(ordered)), gridspec_kw={"width_ratios": [1.75, 1.0]})
    if len(ordered) == 1:
        axes = np.asarray([axes])
    for r, (f, (traj_bgr, map_bgr)) in enumerate(ordered):
        axes[r, 0].imshow(cv2.cvtColor(traj_bgr, cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"Frame {f}: RGB Trajectory & 0.8s Prediction Overlay")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB))
        axes[r, 1].set_title(f"Frame {f}: Camera-Independent 2D Floor Map")
        axes[r, 1].axis("off")
    fig.tight_layout()
    out_path = plots_dir / "stage5_topview_snapshot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_trajectory_and_map_videos(profile_name: str = "local") -> dict[str, Path]:
    """Render trajectory overlay, 2D top-view map, side-by-side comparison, and snapshot plot."""
    CONFIG.paths.ensure_dirs(profile_name)
    s1, _, _, s3, df4, preds = _load_stage_data(profile_name)
    K = CONFIG.camera.K

    cap = cv2.VideoCapture(str(CONFIG.paths.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video {CONFIG.paths.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CONFIG.camera.width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CONFIG.camera.height
    fps = float(cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.fps)

    out_dir = CONFIG.paths.outputs_dir
    p_traj = out_dir / "2_trajectory_overlay.mp4"
    p_map = out_dir / "3_topview_2d_map.mp4"
    p_sbs = out_dir / "3b_side_by_side_map.mp4"

    wr_traj = _open_video_writer(p_traj, w, h, fps)
    wr_map = _open_video_writer(p_map, h, h, fps)
    wr_sbs = _open_video_writer(p_sbs, w + h, h, fps)

    renderer = TopViewMapRenderer(s3, df4, preds, width=h, height=h)

    # Pick representative frames from active dynamic segments for report snapshot.
    snap_targets = {min(185, len(df4) - 1), min(420, len(df4) - 1)}
    snap_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    f = 0
    while f < len(df4):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        traj_frame = _draw_rgb_trajectory_frame(frame_bgr, f, s1, s3, df4, preds, K)
        map_frame = renderer.render_frame(f)
        sbs_frame = np.hstack([traj_frame, map_frame])
        cv2.line(sbs_frame, (w, 0), (w, h - 1), (180, 185, 195), 2)

        wr_traj.write(traj_frame)
        wr_map.write(map_frame)
        wr_sbs.write(sbs_frame)
        if f in snap_targets:
            snap_frames[f] = (traj_frame, map_frame)
        f += 1

    cap.release()
    wr_traj.release()
    wr_map.release()
    wr_sbs.release()

    p_snap = _save_snapshot_figure(snap_frames, CONFIG.paths.plots_dir)
    return {
        "trajectory_video": p_traj,
        "topview_video": p_map,
        "side_by_side_video": p_sbs,
        "snapshot_plot": p_snap,
    }


def _style_worksheet(ws, num_formats: dict[int, str] | None = None, freeze_cell: str = "A2", add_filter: bool = False) -> None:
    header_fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    thin = Side(border_style="thin", color="D9E1F2")
    cell_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = cell_border

    ws.freeze_panes = freeze_cell
    if add_filter and ws.max_row >= 1 and ws.max_column >= 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"

    for r_idx in range(2, ws.max_row + 1):
        for c_idx in range(1, ws.max_column + 1):
            c = ws.cell(row=r_idx, column=c_idx)
            c.border = cell_border
            if num_formats and c_idx in num_formats and isinstance(c.value, (int, float)) and np.isfinite(c.value):
                c.number_format = num_formats[c_idx]

    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max((len(str(cell.value)) if cell.value is not None else 0) for cell in col)
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 10), 42)


def export_telemetry_workbook(profile_name: str = "local") -> tuple[Path, Path]:
    """Export 3-sheet Excel telemetry workbook and flat CSV matching Sheet 2."""
    CONFIG.paths.ensure_dirs(profile_name)
    s1, df2, _, s3, df4, _ = _load_stage_data(profile_name)
    prof = CONFIG.get_profile(profile_name)

    det_cnt = int(s1.detected.sum())
    g0, g1 = CONFIG.video.gap_frames
    gap_fp = int(s1.detected[g0 : min(g1 + 1, len(s1.detected))].sum()) if len(s1.detected) > g0 else 0
    pose_cnt = int(s3.pose_valid.sum())
    stat_ok = df4[(df4["track_status"] != "gap") & (df4["p_stationary"] > 0.7)]
    if len(stat_ok) >= 2:
        stat_std = stat_ok[["X_g", "Y_g"]].to_numpy().std(axis=0) * 100.0
        stat_val = f"{stat_std[0]:.2f}, {stat_std[1]:.2f}"
        stat_note = f"World-frame stability over F{int(stat_ok['frame_idx'].min())}-F{int(stat_ok['frame_idx'].max())}"
    else:
        stat_val = "N/A"
        stat_note = "No sustained stationary segment detected"

    errs_all = df4["reproj_err_px"].to_numpy(dtype=np.float64)
    errs_all_v = errs_all[np.isfinite(errs_all)]
    errs_unclip = errs_all[np.isfinite(errs_all) & (~s1.border_clipped)]

    summary_rows = [
        ("Active Profile", prof.name, "-", "Local CPU pipeline vs. optional Cloud GPU hooks"),
        ("2D Detector Backend", prof.detector, "-", "YOLO11-seg ROI/tiled + two-tier Lab/HSV ellipse fit"),
        ("Ego-Motion Backend", prof.vo, "-", "Static-masked bundle-adjusted SfM + LK PnP"),
        ("Image Resolution", f"{CONFIG.camera.width}x{CONFIG.camera.height}", "px", "Intel RealSense D435i RGB stream"),
        ("Frame Rate / Count", f"{CONFIG.video.fps:.1f} / {len(df4)}", "fps / frames", "15.40 s duration"),
        ("Camera HFOV / VFOV", f"{CONFIG.camera.hfov_deg:.1f} / {CONFIG.camera.vfov_deg:.1f}", "deg", "Zero nominal Brown-Conrady distortion"),
        ("Focal Length (fx, fy)", f"{CONFIG.camera.fx:.2f}, {CONFIG.camera.fy:.2f}", "px", f"Principal point ({CONFIG.camera.cx:.1f}, {CONFIG.camera.cy:.1f})"),
        ("Football Spec (FIFA 5)", f"D={CONFIG.ball.diameter_m:.2f}, R={CONFIG.ball.radius_m:.2f}, m={CONFIG.ball.mass_kg:.2f}", "m, m, kg", f"Rolling mu_r={CONFIG.ball.rolling_friction:.2f}, Cd={CONFIG.ball.drag_coeff:.2f}"),
        ("Detected Ball Frames", f"{det_cnt}/{len(df4)}", "frames", f"Gap F{g0}-F{g1} false positives: {gap_fp}"),
        ("Localized Camera Poses", f"{pose_cnt}/{len(df4)}", "frames", "Keyframes + bidirectional LK PnP"),
        ("Metric Scale Anchor (alpha)", round(float(s3.scale_alpha), 6), "m/unit", "DLT stationary ball + soft_l1 alignment"),
        ("Median Camera Height (C_z_g)", round(float(np.median(s3.C_g[:, 2])), 3), "m", "Above RANSAC terrace ground plane"),
        ("Pre-kick Stationary Std (X_g, Y_g)", stat_val, "cm", stat_note),
        ("Reprojection Error (Unclipped)", f"med={np.median(errs_unclip):.2f}, p95={np.percentile(errs_unclip, 95):.2f}", "px", "2D consistency excluding border-clipped caps"),
        ("Reprojection Error (All Observed)", f"med={np.median(errs_all_v):.2f}, p95={np.percentile(errs_all_v, 95):.2f}", "px", "Includes 10 partial border-clipped frames"),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Parameter", "Value", "Units", "Notes"])

    df_frames = pd.DataFrame(
        {
            "frame_idx": df4["frame_idx"].astype(int),
            "timestamp_s": df4["timestamp_s"].astype(float),
            "segment_id": df4["segment_id"].astype(int),
            "track_status": df4["track_status"],
            "det_source": s1.det_source,
            "X_c_m": df4["X_c_smooth"],
            "Y_c_m": df4["Y_c_smooth"],
            "Z_c_m": df4["Z_c_smooth"],
            "distance_m": df4["distance_smooth_m"],
            "Vx_c_mps": df4["Vx_c_rel"],
            "Vy_c_mps": df4["Vy_c_rel"],
            "Vz_c_mps": df4["Vz_c_rel"],
            "speed_c_mps": df4["speed_c_rel"],
            "X_g_m": df4["X_g"],
            "Y_g_m": df4["Y_g"],
            "Z_g_m": df4["Z_g"],
            "Vx_g_mps": df4["Vx_g"],
            "Vy_g_mps": df4["Vy_g"],
            "Vz_g_mps": df4["Vz_g"],
            "speed_g_mps": df4["speed_g"],
            "camera_Xg_m": df4["C_x_g"],
            "camera_Yg_m": df4["C_y_g"],
            "camera_Zg_m": df4["C_z_g"],
            "sigma_Z_m": df2["sigma_Z"],
            "p_stationary": df4["p_stationary"],
            "p_rolling": df4["p_rolling"],
            "p_airborne": df4["p_airborne"],
            "reproj_err_px": df4["reproj_err_px"],
        }
    )

    seg_rows = []
    for sid in sorted(int(s) for s in df4["segment_id"].unique() if int(s) >= 0):
        sub = df4[df4["segment_id"] == sid].sort_values("frame_idx")
        f_a, f_b = int(sub["frame_idx"].iloc[0]), int(sub["frame_idx"].iloc[-1])
        xy = sub[["X_g", "Y_g"]].to_numpy(dtype=np.float64)
        dist_traveled = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) >= 2 else 0.0
        p_means = [float(sub["p_stationary"].mean()), float(sub["p_rolling"].mean()), float(sub["p_airborne"].mean())]
        dom_mode = ["STATIONARY", "ROLLING", "AIRBORNE"][int(np.argmax(p_means))]
        sub_err = sub["reproj_err_px"].to_numpy(dtype=np.float64)
        unclip_mask = ~s1.border_clipped[sub["frame_idx"].to_numpy(dtype=int)]
        sub_err_u = sub_err[np.isfinite(sub_err) & unclip_mask]
        seg_rows.append(
            {
                "segment_id": sid,
                "start_frame": f_a,
                "end_frame": f_b,
                "num_frames": len(sub),
                "duration_s": float(sub["timestamp_s"].iloc[-1] - sub["timestamp_s"].iloc[0]),
                "initial_speed_mps": float(sub["speed_g"].iloc[0]),
                "peak_speed_mps": float(sub["speed_g"].max()),
                "median_speed_mps": float(sub["speed_g"].median()),
                "mean_height_Zg_m": float(sub["Z_g"].mean()),
                "ground_distance_m": dist_traveled,
                "median_unclip_reproj_px": float(np.median(sub_err_u)) if len(sub_err_u) else float("nan"),
                "dominant_mode": dom_mode,
            }
        )
    df_segments = pd.DataFrame(seg_rows)

    out_xlsx = CONFIG.paths.outputs_dir / "ball_3d_telemetry.xlsx"
    out_csv = CONFIG.paths.outputs_dir / "ball_3d_telemetry.csv"
    df_frames.to_csv(out_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary & Calibration", index=False)
        df_frames.to_excel(writer, sheet_name="Frame Telemetry", index=False)
        df_segments.to_excel(writer, sheet_name="Segment Summary", index=False)

    wb = openpyxl.load_workbook(out_xlsx)
    _style_worksheet(wb["Summary & Calibration"], num_formats={2: "0.000"}, freeze_cell="A2")
    frame_fmts = {i: "0.000" for i in range(6, 28)}
    frame_fmts[1] = "0"
    frame_fmts[2] = "0.000"
    frame_fmts[3] = "0"
    frame_fmts[28] = "0.00"
    _style_worksheet(wb["Frame Telemetry"], num_formats=frame_fmts, freeze_cell="A2", add_filter=True)
    seg_fmts = {1: "0", 2: "0", 3: "0", 4: "0", 5: "0.000", 6: "0.00", 7: "0.00", 8: "0.00", 9: "0.000", 10: "0.00", 11: "0.00"}
    _style_worksheet(wb["Segment Summary"], num_formats=seg_fmts, freeze_cell="A2")
    wb.save(out_xlsx)
    return out_xlsx, out_csv


def run_stage5(profile_name: str = "local") -> dict[str, Path]:
    """Generate final telemetry workbooks and trajectory/top-view deliverables."""
    xlsx_path, csv_path = export_telemetry_workbook(profile_name)
    vid_paths = render_trajectory_and_map_videos(profile_name)
    return {"xlsx": xlsx_path, "csv": csv_path, **vid_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: Render deliverables and export telemetry.")
    parser.add_argument("--profile", default="local", choices=("local", "gpu"))
    args = parser.parse_args()
    out = run_stage5(args.profile)
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
