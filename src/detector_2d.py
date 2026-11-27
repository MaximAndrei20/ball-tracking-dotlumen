# Stage 1 2D ball detection. Includes YOLO ellipse fitting, motion blur fallback, static masks, and SAM2/SAM3 GPU hooks.

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import CONFIG, ProfileConfig

_DILATE_5PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
_DILATE_20PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
_MORPH_3PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class FrameDetection:
    """Per-frame 2D ball detection and reliability metadata."""

    frame_idx: int
    timestamp_s: float
    detected: bool
    det_source: str
    bbox_xyxy: np.ndarray
    center_uv: np.ndarray
    semi_axes_ab: np.ndarray
    ellipse_angle_deg: float
    mask_area_px: float
    conf: float
    is_blurred: bool
    occluded_person: bool
    border_clipped: bool


@dataclass
class Stage1Result:
    """Serialized Stage 1 detections and bit-packed static scene masks."""

    frame_idx: np.ndarray
    timestamp_s: np.ndarray
    detected: np.ndarray
    det_source: np.ndarray
    bbox_xyxy: np.ndarray
    center_uv: np.ndarray
    semi_axes_ab: np.ndarray
    ellipse_angle_deg: np.ndarray
    mask_area_px: np.ndarray
    conf: np.ndarray
    is_blurred: np.ndarray
    occluded_person: np.ndarray
    border_clipped: np.ndarray
    static_masks_packed: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame_idx": self.frame_idx,
                "timestamp_s": self.timestamp_s,
                "detected": self.detected,
                "det_source": self.det_source,
                "x1": self.bbox_xyxy[:, 0],
                "y1": self.bbox_xyxy[:, 1],
                "x2": self.bbox_xyxy[:, 2],
                "y2": self.bbox_xyxy[:, 3],
                "u": self.center_uv[:, 0],
                "v": self.center_uv[:, 1],
                "a_px": self.semi_axes_ab[:, 0],
                "b_px": self.semi_axes_ab[:, 1],
                "ellipse_angle_deg": self.ellipse_angle_deg,
                "mask_area_px": self.mask_area_px,
                "conf": self.conf,
                "is_blurred": self.is_blurred,
                "occluded_person": self.occluded_person,
                "border_clipped": self.border_clipped,
            }
        )

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / "stage1_2d.npz",
            frame_idx=self.frame_idx,
            timestamp_s=self.timestamp_s,
            detected=self.detected,
            det_source=self.det_source,
            bbox_xyxy=self.bbox_xyxy,
            center_uv=self.center_uv,
            semi_axes_ab=self.semi_axes_ab,
            ellipse_angle_deg=self.ellipse_angle_deg,
            mask_area_px=self.mask_area_px,
            conf=self.conf,
            is_blurred=self.is_blurred,
            occluded_person=self.occluded_person,
            border_clipped=self.border_clipped,
            static_masks_packed=self.static_masks_packed,
        )
        self.to_dataframe().to_csv(cache_dir / "stage1_2d.csv", index=False)

    @classmethod
    def load(cls, cache_dir: Path) -> Stage1Result:
        with np.load(cache_dir / "stage1_2d.npz", allow_pickle=False) as d:
            return cls(
                frame_idx=d["frame_idx"],
                timestamp_s=d["timestamp_s"],
                detected=d["detected"],
                det_source=d["det_source"],
                bbox_xyxy=d["bbox_xyxy"],
                center_uv=d["center_uv"],
                semi_axes_ab=d["semi_axes_ab"],
                ellipse_angle_deg=d["ellipse_angle_deg"],
                mask_area_px=d["mask_area_px"],
                conf=d["conf"],
                is_blurred=d["is_blurred"],
                occluded_person=d["occluded_person"],
                border_clipped=d["border_clipped"],
                static_masks_packed=d["static_masks_packed"],
            )

    @staticmethod
    def unpack_static_mask(
        packed_row: np.ndarray,
        height: int = CONFIG.camera.height,
        width: int = CONFIG.camera.width,
    ) -> np.ndarray:
        return np.unpackbits(packed_row)[: height * width].reshape(height, width).astype(bool)


class BaseDetector(ABC):
    """Abstract interface for Stage 1 2D ball detectors and foreground maskers."""

    @abstractmethod
    def run(self, video_path: Path, max_frames: int | None = None) -> Stage1Result:
        """Process video frames and return Stage 1 detections and static masks."""


class _Kalman2D:
    """Constant-velocity 2D image-space Kalman gate for outlier and reflection rejection."""

    def __init__(self, max_missed: int = 6) -> None:
        self.max_missed = max_missed
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag([4.0, 4.0, 64.0, 64.0])
        self.R = np.diag([9.0, 9.0])
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.missed_frames = 0
        self.expected_diam = 0.0
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 50.0

    def predict(self) -> tuple[float, float] | None:
        if not self.active:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missed_frames += 1
        return float(self.x[0]), float(self.x[1])

    def mahalanobis_sq(self, u: float, v: float) -> float:
        innov = np.array([u, v], dtype=np.float64) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return float(innov @ np.linalg.solve(S, innov))

    def gate(self, u: float, v: float, diam: float, clipped: bool) -> bool:
        if not self.active:
            return True
        dist = float(np.hypot(u - self.x[0], v - self.x[1]))
        max_jump = max(120.0, 2.5 * self.expected_diam)
        if self.mahalanobis_sq(u, v) > 30.0 or dist > max_jump:
            return False
        if not clipped and self.expected_diam > 0.0:
            ratio = diam / self.expected_diam
            if ratio < 0.55 or ratio > 1.85:
                return False
        return True

    def update(self, u: float, v: float, diam: float, clipped: bool) -> None:
        z = np.array([u, v], dtype=np.float64)
        if not self.active:
            self.x = np.array([u, v, 0.0, 0.0], dtype=np.float64)
            self.P = np.diag([16.0, 16.0, 64.0, 64.0])
            self.active = True
            self.missed_frames = 0
            if not clipped:
                self.expected_diam = diam
            return

        innov = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed_frames = 0
        if not clipped:
            self.expected_diam = diam if self.expected_diam <= 0.0 else 0.7 * self.expected_diam + 0.3 * diam

    def mark_missed(self) -> None:
        if self.active and self.missed_frames > self.max_missed:
            self.reset()


def _core_pink_mask(bgr: np.ndarray) -> np.ndarray:
    """High-precision magenta core mask excluding dark maroon clothing and orange roofs."""
    vcfg = CONFIG.video
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (
        (lab[:, :, 1] >= vcfg.ball_core_lab_a_min)
        & (lab[:, :, 0] >= vcfg.ball_core_lab_l_min)
        & (hsv[:, :, 0] >= vcfg.ball_core_hsv_h_min)
        & (hsv[:, :, 1] >= vcfg.ball_core_hsv_s_min)
        & (hsv[:, :, 2] >= vcfg.ball_core_hsv_v_min)
    )


def _refine_chroma_mask(bgr: np.ndarray) -> np.ndarray:
    """Chroma-only mask preserving self-shadowed lower ball pixels while excluding grey tiles."""
    vcfg = CONFIG.video
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_ok = (hsv[:, :, 0] >= vcfg.ball_refine_hsv_h_high_min) | (hsv[:, :, 0] <= vcfg.ball_refine_hsv_h_low_max)
    return (lab[:, :, 1] >= vcfg.ball_refine_lab_a_min) & (hsv[:, :, 1] >= vcfg.ball_refine_hsv_s_min) & hue_ok


def _extract_person_mask(yolo_res, height: int, width: int) -> np.ndarray:
    """Build full-frame boolean person mask from Ultralytics segmentation polygons."""
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = yolo_res.boxes
    if boxes is None or len(boxes) == 0:
        return mask.astype(bool)

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    polys = yolo_res.masks.xy if yolo_res.masks is not None else None

    for i, cid in enumerate(cls_ids):
        if cid != 0:
            continue
        if polys is not None and len(polys[i]) >= 3:
            cv2.fillPoly(mask, [polys[i].astype(np.int32)], 1)
        else:
            x1, y1, x2, y2 = np.clip(xyxy[i].astype(int), [0, 0, 0, 0], [width, height, width, height])
            mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def _extract_yolo_ball_boxes(yolo_res, offset_xy: tuple[int, int] = (0, 0)) -> list[tuple[float, float, float, float, float]]:
    """Extract COCO sports-ball boxes shifted by ROI offset."""
    boxes = yolo_res.boxes
    if boxes is None or len(boxes) == 0:
        return []
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    ox, oy = offset_xy
    out = []
    for i, cid in enumerate(cls_ids):
        if cid == 32:
            x1, y1, x2, y2 = xyxy[i]
            out.append((float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy), float(confs[i])))
    return out


def _propose_chroma_candidates(
    frame: np.ndarray,
    person_mask: np.ndarray,
    border_only: bool = False,
) -> list[tuple[float, float, float, float, float]]:
    """Propose bright-pink floor blobs outside the person mask and glass reflection zone."""
    h, w = frame.shape[:2]
    core = _core_pink_mask(frame) & (~person_mask)
    core[:145, :] = False
    if border_only:
        core[: h - 70, :] = False
    else:
        # Exclude uninitialized locks on the dim reflection strip inside the right glass doors.
        core[:, 1120:] = False

    mask_u8 = cv2.morphologyEx(core.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _MORPH_3PX)
    num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    min_area = 120 if border_only else 45
    proposals = []
    for lbl in range(1, num_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < min_area or max(bw, bh) > 150:
            continue
        if border_only and (y + bh < h - 5):
            continue
        if not border_only and max(bw, bh) / max(1, min(bw, bh)) > 2.6:
            continue
        proposals.append((float(x), float(y), float(x + bw), float(y + bh), 0.35 if border_only else 0.50))
    proposals.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return proposals[:3]


def _refine_ball_ellipse(
    frame: np.ndarray,
    person_mask: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    conf: float,
    source: str,
) -> tuple[dict, np.ndarray] | None:
    """Refine a ball candidate via chroma-only masking and Fitzgibbon direct ellipse fitting."""
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = box_xyxy
    cx0, cy0 = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
    ix0, iy0 = int(np.clip(round(cx0), 0, w - 1)), int(np.clip(round(cy0), 0, h - 1))

    if cy0 < 145.0 or person_mask[iy0, ix0]:
        return None

    half = max(28.0, 0.9 * max(bx2 - bx1, by2 - by1))
    rx1 = max(0, int(np.floor(cx0 - half)))
    ry1 = max(0, int(np.floor(cy0 - half)))
    rx2 = min(w, int(np.ceil(cx0 + half)))
    ry2 = min(h, int(np.ceil(cy0 + half)))
    if rx2 - rx1 < 12 or ry2 - ry1 < 12:
        return None

    roi = frame[ry1:ry2, rx1:rx2]
    p_roi = person_mask[ry1:ry2, rx1:rx2]
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    mask = _refine_chroma_mask(roi)
    # Keep dark maroon shirt pixels out if person overlaps ROI; black shoes already fail chroma.
    mask &= ~(p_roi & (lab_roi[:, :, 0] < 65))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _MORPH_3PX)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, _MORPH_3PX)

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    target_pt = np.array([cx0 - rx1, cy0 - ry1], dtype=np.float64)
    best_cnt = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 45.0:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            continue
        mx, my = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dist = float(np.hypot(mx - target_pt[0], my - target_pt[1]))
        if dist > max(26.0, 0.7 * max(bx2 - bx1, by2 - by1)):
            continue
        score = area / (1.0 + 0.15 * dist)
        if score > best_score:
            best_score = score
            best_cnt = c

    if best_cnt is None:
        return None

    area = float(cv2.contourArea(best_cnt))
    x, y, bw, bh = cv2.boundingRect(best_cnt)
    tx1, ty1, tx2, ty2 = float(rx1 + x), float(ry1 + y), float(rx1 + x + bw), float(ry1 + y + bh)
    clipped = (tx1 <= 10.0) or (ty1 <= 10.0) or (tx2 >= w - 11.0) or (ty2 >= h - 11.0)

    core_roi = _core_pink_mask(roi)
    cnt_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(cnt_mask, [best_cnt], -1, 1, thickness=-1)
    core_frac = float(np.sum(core_roi & (cnt_mask > 0))) / max(area, 1.0)
    if core_frac < 0.15:
        return None

    if len(best_cnt) >= 5:
        try:
            (ex, ey), (d1, d2), ang = cv2.fitEllipseAMS(best_cnt)
        except cv2.error:
            (ex, ey), (d1, d2), ang = cv2.fitEllipse(best_cnt)
        a, b = 0.5 * max(d1, d2), 0.5 * min(d1, d2)
        if d1 < d2:
            ang = (ang + 90.0) % 180.0
    else:
        ex, ey = x + 0.5 * bw, y + 0.5 * bh
        a, b, ang = 0.5 * max(bw, bh), 0.5 * min(bw, bh), 0.0

    if not np.isfinite(a) or not np.isfinite(b) or b < 3.0 or (not clipped and a / b > 2.0):
        return None

    full_contour = best_cnt + np.array([[[rx1, ry1]]], dtype=best_cnt.dtype)
    info = {
        "det_source": source,
        "bbox_xyxy": np.array([tx1, ty1, tx2, ty2], dtype=np.float32),
        "center_uv": np.array([rx1 + ex, ry1 + ey], dtype=np.float64),
        "semi_axes_ab": np.array([a, b], dtype=np.float64),
        "ellipse_angle_deg": float(ang),
        "mask_area_px": area,
        "conf": float(min(1.0, 0.7 * conf + 0.3 * min(1.0, core_frac * 2.0))),
        "border_clipped": bool(clipped),
    }
    return info, full_contour


def _framediff_fallback(
    prev_f: np.ndarray,
    curr_f: np.ndarray,
    next_f: np.ndarray,
    person_mask: np.ndarray,
    pred_uv: tuple[float, float],
    exp_diam: float,
) -> tuple[dict, np.ndarray] | None:
    """Ego-motion-compensated 3-frame differencing fallback inside Kalman ROI."""
    h, w = curr_f.shape[:2]
    prev_g = cv2.cvtColor(prev_f, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_f, cv2.COLOR_BGR2GRAY)
    next_g = cv2.cvtColor(next_f, cv2.COLOR_BGR2GRAY)

    bg_mask = (~person_mask).astype(np.uint8) * 255
    pts_curr = cv2.goodFeaturesToTrack(curr_g, maxCorners=150, qualityLevel=0.05, minDistance=20, mask=bg_mask)
    if pts_curr is None or len(pts_curr) < 12:
        return None

    def _warp_to_curr(src_bgr: np.ndarray, src_g: np.ndarray) -> np.ndarray | None:
        pts_src, st, _ = cv2.calcOpticalFlowPyrLK(curr_g, src_g, pts_curr, None)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            return None
        H_mat, _ = cv2.findHomography(pts_src[ok], pts_curr[ok], cv2.USAC_MAGSAC, 2.0)
        if H_mat is None:
            return None
        return cv2.warpPerspective(src_bgr, H_mat, (w, h))

    w_prev = _warp_to_curr(prev_f, prev_g)
    w_next = _warp_to_curr(next_f, next_g)
    if w_prev is None or w_next is None:
        return None

    half = int(max(60.0, 2.0 * max(exp_diam, 30.0)))
    x1 = max(0, int(pred_uv[0] - half))
    y1 = max(0, int(pred_uv[1] - half))
    x2 = min(w, int(pred_uv[0] + half))
    y2 = min(h, int(pred_uv[1] + half))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    d_prev = np.max(cv2.absdiff(curr_f[y1:y2, x1:x2], w_prev[y1:y2, x1:x2]), axis=2) > 20
    d_next = np.max(cv2.absdiff(curr_f[y1:y2, x1:x2], w_next[y1:y2, x1:x2]), axis=2) > 20
    motion_pink = d_prev & d_next & _refine_chroma_mask(curr_f[y1:y2, x1:x2]) & (~person_mask[y1:y2, x1:x2])
    m_u8 = cv2.morphologyEx(motion_pink.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _MORPH_3PX)
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 40.0:
        return None
    bx, by, bw, bh = cv2.boundingRect(best)
    return _refine_ball_ellipse(curr_f, person_mask, (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh), 0.30, "framediff")


def _run_roi_yolo(
    model: YOLO,
    device: str,
    frame: np.ndarray,
    center_uv: tuple[float, float],
    crop_size: int = 320,
) -> list[tuple[float, float, float, float, float]]:
    h, w = frame.shape[:2]
    half = crop_size // 2
    x1 = int(np.clip(round(center_uv[0] - half), 0, max(0, w - crop_size)))
    y1 = int(np.clip(round(center_uv[1] - half), 0, max(0, h - crop_size)))
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    res = model.predict(frame[y1:y2, x1:x2], imgsz=crop_size, classes=[32], conf=0.10, device=device, verbose=False)[0]
    return _extract_yolo_ball_boxes(res, offset_xy=(x1, y1))


def _collect_ball_candidates(
    model: YOLO,
    device: str,
    frame: np.ndarray,
    person_mask: np.ndarray,
    yolo_res,
    tracker: _Kalman2D,
    pred_uv: tuple[float, float] | None,
    next_frame: np.ndarray | None,
) -> list[tuple[float, float, float, float, float, str]]:
    cands: list[tuple[float, float, float, float, float, str]] = [
        (*b, "yolo_tile") for b in _extract_yolo_ball_boxes(yolo_res)
    ]

    if pred_uv is not None:
        near_full = any(np.hypot(0.5 * (b[0] + b[2]) - pred_uv[0], 0.5 * (b[1] + b[3]) - pred_uv[1]) < 80.0 for b in cands)
        if not near_full:
            cands.extend((*b, "yolo_roi") for b in _run_roi_yolo(model, device, frame, pred_uv))

    if not cands:
        for pb in _propose_chroma_candidates(frame, person_mask, border_only=False):
            roi_boxes = _run_roi_yolo(model, device, frame, (0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])))
            if roi_boxes:
                cands.extend((*b, "yolo_tile") for b in roi_boxes)
            elif tracker.active:
                cands.append((*pb, "chroma_border"))

    # Confirm bottom-border clipped appearances (exit F205-206 or re-entry F380-382) when YOLO misses half-cut balls.
    for bb in _propose_chroma_candidates(frame, person_mask, border_only=True):
        bcx, bcy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
        confirmed = pred_uv is not None and pred_uv[1] > frame.shape[0] - 110
        if not confirmed and next_frame is not None:
            nx = int(np.clip(round(bcx), 0, frame.shape[1] - 1))
            ny = max(0, int(round(bcy - 30)))
            patch = next_frame[max(0, ny - 55) : min(frame.shape[0], ny + 65), max(0, nx - 65) : min(frame.shape[1], nx + 65)]
            confirmed = int(np.sum(_core_pink_mask(patch))) >= 120
        if confirmed:
            cands.append((*bb, "chroma_border"))

    return cands


class YoloEllipseDetector(BaseDetector):
    """YOLO11-seg + native ROI/chroma ellipse fitting + 3-frame differencing fallback."""

    def __init__(self, weights: str = "yolo11s-seg.pt", device: str = "cpu") -> None:
        self.model = YOLO(weights)
        self.device = device

    def run(self, video_path: Path, max_frames: int | None = None) -> Stage1Result:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = min(total, max_frames) if max_frames is not None else total
        h, w = CONFIG.camera.height, CONFIG.camera.width

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")

        tracker = _Kalman2D(max_missed=6)
        records: list[FrameDetection] = []
        packed_masks: list[np.ndarray] = []
        prev_frame: np.ndarray | None = None
        curr_frame: np.ndarray | None = first_frame

        for idx in range(n_frames):
            next_frame = cap.read()[1] if idx + 1 < n_frames else None
            assert curr_frame is not None

            yolo_res = self.model.predict(curr_frame, imgsz=640, classes=[0, 32], conf=0.10, device=self.device, verbose=False)[0]
            person_mask = _extract_person_mask(yolo_res, h, w)
            pred_uv = tracker.predict()

            raw_cands = _collect_ball_candidates(self.model, self.device, curr_frame, person_mask, yolo_res, tracker, pred_uv, next_frame)
            winner: tuple[dict, np.ndarray] | None = None
            best_score = -1.0

            for x1, y1, x2, y2, conf, src in raw_cands:
                refined = _refine_ball_ellipse(curr_frame, person_mask, (x1, y1, x2, y2), conf, src)
                if refined is None:
                    continue
                info, contour = refined
                u, v = float(info["center_uv"][0]), float(info["center_uv"][1])
                diam = float(2.0 * info["semi_axes_ab"][1])
                if not tracker.gate(u, v, diam, info["border_clipped"]):
                    continue
                dist_pen = 0.0 if pred_uv is None else 0.01 * float(np.hypot(u - pred_uv[0], v - pred_uv[1]))
                score = info["conf"] - dist_pen
                if score > best_score:
                    best_score = score
                    winner = (info, contour)

            if winner is None and tracker.active and pred_uv is not None and prev_frame is not None and next_frame is not None:
                fd_res = _framediff_fallback(prev_frame, curr_frame, next_frame, person_mask, pred_uv, tracker.expected_diam)
                if fd_res is not None:
                    info, _ = fd_res
                    if tracker.gate(float(info["center_uv"][0]), float(info["center_uv"][1]), float(2.0 * info["semi_axes_ab"][1]), info["border_clipped"]):
                        winner = fd_res

            ball_mask = np.zeros((h, w), dtype=bool)
            if winner is not None:
                info, contour = winner
                u, v = float(info["center_uv"][0]), float(info["center_uv"][1])
                a, b = float(info["semi_axes_ab"][0]), float(info["semi_axes_ab"][1])
                tracker.update(u, v, 2.0 * b, info["border_clipped"])

                b_u8 = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(b_u8, [contour], -1, 1, thickness=-1)
                ball_mask = b_u8.astype(bool)
                person_dil5 = cv2.dilate(person_mask.astype(np.uint8), _DILATE_5PX).astype(bool)
                occluded = bool(np.any(person_dil5 & ball_mask))
                blurred = (not info["border_clipped"]) and (a / max(b, 1e-6) > 1.25)

                records.append(
                    FrameDetection(
                        frame_idx=idx,
                        timestamp_s=idx / fps,
                        detected=True,
                        det_source=info["det_source"],
                        bbox_xyxy=info["bbox_xyxy"],
                        center_uv=info["center_uv"],
                        semi_axes_ab=info["semi_axes_ab"],
                        ellipse_angle_deg=info["ellipse_angle_deg"],
                        mask_area_px=info["mask_area_px"],
                        conf=info["conf"],
                        is_blurred=blurred,
                        occluded_person=occluded,
                        border_clipped=info["border_clipped"],
                    )
                )
            else:
                tracker.mark_missed()
                records.append(
                    FrameDetection(
                        frame_idx=idx,
                        timestamp_s=idx / fps,
                        detected=False,
                        det_source="none",
                        bbox_xyxy=np.full(4, np.nan, dtype=np.float32),
                        center_uv=np.full(2, np.nan, dtype=np.float64),
                        semi_axes_ab=np.full(2, np.nan, dtype=np.float64),
                        ellipse_angle_deg=float("nan"),
                        mask_area_px=float("nan"),
                        conf=0.0,
                        is_blurred=False,
                        occluded_person=False,
                        border_clipped=False,
                    )
                )

            fg = (person_mask | ball_mask).astype(np.uint8)
            static_mask = ~cv2.dilate(fg, _DILATE_20PX).astype(bool)
            packed_masks.append(np.packbits(static_mask.reshape(-1)))

            prev_frame = curr_frame
            curr_frame = next_frame

        cap.release()
        return _pack_stage1_result(records, packed_masks)


def _pack_stage1_result(records: list[FrameDetection], packed_masks: list[np.ndarray]) -> Stage1Result:
    return Stage1Result(
        frame_idx=np.array([r.frame_idx for r in records], dtype=np.int32),
        timestamp_s=np.array([r.timestamp_s for r in records], dtype=np.float64),
        detected=np.array([r.detected for r in records], dtype=bool),
        det_source=np.array([r.det_source for r in records], dtype=str),
        bbox_xyxy=np.vstack([r.bbox_xyxy for r in records]).astype(np.float32),
        center_uv=np.vstack([r.center_uv for r in records]).astype(np.float64),
        semi_axes_ab=np.vstack([r.semi_axes_ab for r in records]).astype(np.float64),
        ellipse_angle_deg=np.array([r.ellipse_angle_deg for r in records], dtype=np.float64),
        mask_area_px=np.array([r.mask_area_px for r in records], dtype=np.float64),
        conf=np.array([r.conf for r in records], dtype=np.float32),
        is_blurred=np.array([r.is_blurred for r in records], dtype=bool),
        occluded_person=np.array([r.occluded_person for r in records], dtype=bool),
        border_clipped=np.array([r.border_clipped for r in records], dtype=bool),
        static_masks_packed=np.vstack(packed_masks).astype(np.uint8),
    )


def _empty_frame_detection(frame_idx: int, timestamp_s: float) -> FrameDetection:
    return FrameDetection(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        detected=False,
        det_source="none",
        bbox_xyxy=np.full(4, np.nan, dtype=np.float32),
        center_uv=np.full(2, np.nan, dtype=np.float64),
        semi_axes_ab=np.full(2, np.nan, dtype=np.float64),
        ellipse_angle_deg=float("nan"),
        mask_area_px=float("nan"),
        conf=0.0,
        is_blurred=False,
        occluded_person=False,
        border_clipped=False,
    )


def _extract_ball_record_from_mask(
    frame_idx: int,
    timestamp_s: float,
    ball_mask: np.ndarray | None,
    person_mask: np.ndarray,
    conf: float,
    source: str,
    frame_shape: tuple[int, int],
) -> tuple[FrameDetection, np.ndarray]:
    """Convert binary ball and person masks into a FrameDetection and bit-packed static mask."""
    h, w = frame_shape
    p_mask = person_mask.astype(bool)
    b_u8 = np.zeros((h, w), dtype=np.uint8)
    valid_fit = False

    if ball_mask is not None:
        raw_u8 = (ball_mask > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(raw_u8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(best))
            if area >= 45.0:
                x, y, bw, bh = cv2.boundingRect(best)
                tx1, ty1, tx2, ty2 = float(x), float(y), float(x + bw), float(y + bh)
                clipped = (tx1 <= 10.0) or (ty1 <= 10.0) or (tx2 >= w - 11.0) or (ty2 >= h - 11.0)
                if len(best) >= 5:
                    try:
                        (ex, ey), (d1, d2), ang = cv2.fitEllipseAMS(best)
                    except cv2.error:
                        (ex, ey), (d1, d2), ang = cv2.fitEllipse(best)
                    a, b = 0.5 * max(d1, d2), 0.5 * min(d1, d2)
                    if d1 < d2:
                        ang = (ang + 90.0) % 180.0
                else:
                    ex, ey = x + 0.5 * bw, y + 0.5 * bh
                    a, b, ang = 0.5 * max(bw, bh), 0.5 * min(bw, bh), 0.0

                if np.isfinite(a) and np.isfinite(b) and b >= 3.0 and (clipped or a / b <= 2.2):
                    valid_fit = True
                    cv2.drawContours(b_u8, [best], -1, 1, thickness=-1)
                    person_dil5 = cv2.dilate(p_mask.astype(np.uint8), _DILATE_5PX).astype(bool)
                    occluded = bool(np.any(person_dil5 & (b_u8 > 0)))
                    blurred = (not clipped) and (a / b > 1.25)
                    rec = FrameDetection(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        detected=True,
                        det_source=source,
                        bbox_xyxy=np.array([tx1, ty1, tx2, ty2], dtype=np.float32),
                        center_uv=np.array([ex, ey], dtype=np.float64),
                        semi_axes_ab=np.array([a, b], dtype=np.float64),
                        ellipse_angle_deg=float(ang),
                        mask_area_px=area,
                        conf=float(conf),
                        is_blurred=blurred,
                        occluded_person=occluded,
                        border_clipped=clipped,
                    )

    if not valid_fit:
        rec = _empty_frame_detection(frame_idx, timestamp_s)

    fg = (p_mask.astype(np.uint8) | b_u8)
    static_mask = ~cv2.dilate(fg, _DILATE_20PX).astype(bool)
    return rec, np.packbits(static_mask.reshape(-1))


def _ensure_cuda_if_requested(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested GPU device '{device}', but CUDA is not available on this machine. "
                "Run on a CUDA instance or use `--profile local`."
            )


class Sam2Detector(BaseDetector):
    """Optional GPU backend: YOLO/chroma candidate prompting + Ultralytics SAM 2.1 segmentation."""

    def __init__(
        self,
        weights: str = "sam2.1_l.pt",
        yolo_weights: str = "yolo11m-seg.pt",
        device: str = "cuda",
    ) -> None:
        self.weights = weights
        self.yolo_weights = yolo_weights
        self.device = device

    def run(self, video_path: Path, max_frames: int | None = None) -> Stage1Result:
        _ensure_cuda_if_requested(self.device)
        from ultralytics import SAM

        yolo_model = YOLO(self.yolo_weights)
        sam_model = SAM(self.weights)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = min(total, max_frames) if max_frames is not None else total
        h, w = CONFIG.camera.height, CONFIG.camera.width

        ret, curr_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")

        tracker = _Kalman2D(max_missed=6)
        records: list[FrameDetection] = []
        packed_masks: list[np.ndarray] = []
        prev_frame: np.ndarray | None = None

        for idx in range(n_frames):
            next_frame = cap.read()[1] if idx + 1 < n_frames else None
            assert curr_frame is not None

            yolo_res = yolo_model.predict(curr_frame, imgsz=640, classes=[0, 32], conf=0.10, device=self.device, verbose=False)[0]
            person_mask = _extract_person_mask(yolo_res, h, w)
            pred_uv = tracker.predict()

            raw_cands = _collect_ball_candidates(yolo_model, self.device, curr_frame, person_mask, yolo_res, tracker, pred_uv, next_frame)
            winner_box: tuple[float, float, float, float, float] | None = None
            winner_contour: np.ndarray | None = None
            best_score = -1.0

            for x1, y1, x2, y2, conf, src in raw_cands:
                refined = _refine_ball_ellipse(curr_frame, person_mask, (x1, y1, x2, y2), conf, src)
                if refined is None:
                    continue
                info, contour = refined
                u, v = float(info["center_uv"][0]), float(info["center_uv"][1])
                diam = float(2.0 * info["semi_axes_ab"][1])
                if not tracker.gate(u, v, diam, info["border_clipped"]):
                    continue
                dist_pen = 0.0 if pred_uv is None else 0.01 * float(np.hypot(u - pred_uv[0], v - pred_uv[1]))
                score = info["conf"] - dist_pen
                if score > best_score:
                    best_score = score
                    winner_box = (x1, y1, x2, y2, info["conf"])
                    winner_contour = contour

            if winner_box is None and tracker.active and pred_uv is not None and prev_frame is not None and next_frame is not None:
                fd_res = _framediff_fallback(prev_frame, curr_frame, next_frame, person_mask, pred_uv, tracker.expected_diam)
                if fd_res is not None:
                    info, contour = fd_res
                    if tracker.gate(float(info["center_uv"][0]), float(info["center_uv"][1]), float(2.0 * info["semi_axes_ab"][1]), info["border_clipped"]):
                        bx1, by1, bx2, by2 = info["bbox_xyxy"].tolist()
                        winner_box = (bx1, by1, bx2, by2, info["conf"])
                        winner_contour = contour

            ball_mask: np.ndarray | None = None
            ball_conf = 0.0
            if winner_box is not None:
                bx1, by1, bx2, by2, ball_conf = winner_box
                sam_res = sam_model.predict(curr_frame, bboxes=[[bx1, by1, bx2, by2]], device=self.device, verbose=False)[0]
                if sam_res.masks is not None and len(sam_res.masks.data) > 0:
                    masks_np = sam_res.masks.data.cpu().numpy()
                    # Choose mask inside prompt box with largest overlap while respecting skyline filter.
                    ix1, iy1 = max(0, int(bx1)), max(0, int(by1))
                    ix2, iy2 = min(w, int(np.ceil(bx2))), min(h, int(np.ceil(by2)))
                    best_m, best_ov = None, 0
                    for m_raw in masks_np:
                        m_bin = (cv2.resize(m_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5) if m_raw.shape != (h, w) else (m_raw > 0.5)
                        ov = int(np.sum(m_bin[iy1:iy2, ix1:ix2]))
                        if ov > best_ov:
                            best_ov = ov
                            best_m = m_bin
                    ball_mask = best_m
                if ball_mask is None and winner_contour is not None:
                    fb = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(fb, [winner_contour], -1, 1, thickness=-1)
                    ball_mask = fb > 0

            rec, packed = _extract_ball_record_from_mask(idx, idx / fps, ball_mask, person_mask, ball_conf, "sam2", (h, w))
            if rec.detected:
                tracker.update(float(rec.center_uv[0]), float(rec.center_uv[1]), float(2.0 * rec.semi_axes_ab[1]), rec.border_clipped)
            else:
                tracker.mark_missed()
            records.append(rec)
            packed_masks.append(packed)
            prev_frame = curr_frame
            curr_frame = next_frame

        cap.release()
        return _pack_stage1_result(records, packed_masks)


class Sam3Detector(BaseDetector):
    """Optional cloud GPU backend: Ultralytics SAM3VideoSemanticPredictor open-vocabulary tracking."""

    def __init__(
        self,
        weights: str = "sam3.pt",
        device: str = "cuda",
        ball_prompt: str = "pink football",
        person_prompt: str = "person",
    ) -> None:
        self.weights = weights
        self.device = device
        self.ball_prompt = ball_prompt
        self.person_prompt = person_prompt

    def run(self, video_path: Path, max_frames: int | None = None) -> Stage1Result:
        _ensure_cuda_if_requested(self.device)
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
        except ImportError as exc:
            raise RuntimeError("SAM 3 requires Ultralytics >= 8.4 with SAM3VideoSemanticPredictor.") from exc
        if not Path(self.weights).exists():
            raise RuntimeError(
                f"SAM 3 weights '{self.weights}' not found. Accept the Meta SAM License on Hugging Face "
                "and stage weights on your GPU box before running --detector sam3."
            )

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_frames = min(total, max_frames) if max_frames is not None else total
        h, w = CONFIG.camera.height, CONFIG.camera.width

        predictor = SAM3VideoSemanticPredictor(
            overrides={
                "model": self.weights,
                "device": self.device,
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "half": self.device.startswith("cuda"),
                "save": False,
            }
        )
        records: list[FrameDetection] = []
        packed_masks: list[np.ndarray] = []

        for idx, res in enumerate(predictor(source=str(video_path), text=[self.ball_prompt, self.person_prompt], stream=True)):
            if idx >= n_frames:
                break
            person_mask = np.zeros((h, w), dtype=bool)
            best_ball_mask: np.ndarray | None = None
            best_conf = 0.0

            if res.masks is not None and res.boxes is not None and len(res.boxes) > 0:
                masks_np = res.masks.data.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                bin_masks = [
                    (
                        cv2.resize(m_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                        if m_raw.shape != (h, w)
                        else (m_raw > 0.5)
                    )
                    for m_raw in masks_np
                ]
                # Pass 1: accumulate all person masks before scoring ball candidates regardless of tensor order.
                for m_bin, cid in zip(bin_masks, cls_ids):
                    if cid == 1:
                        person_mask |= m_bin
                # Pass 2: select highest-confidence ball mask below skyline and outside person pixels.
                for m_bin, cid, conf in zip(bin_masks, cls_ids, confs):
                    if cid == 0 and float(conf) > best_conf:
                        ys, xs = np.where(m_bin)
                        if len(ys) > 0:
                            cx_m = int(np.clip(round(float(np.mean(xs))), 0, w - 1))
                            cy_m = int(np.clip(round(float(np.mean(ys))), 0, h - 1))
                            if cy_m >= 145 and not person_mask[cy_m, cx_m]:
                                best_conf = float(conf)
                                best_ball_mask = m_bin

            rec, packed = _extract_ball_record_from_mask(idx, idx / fps, best_ball_mask, person_mask, best_conf, "sam3", (h, w))
            records.append(rec)
            packed_masks.append(packed)

        return _pack_stage1_result(records, packed_masks)


def create_detector(name: str, profile: ProfileConfig) -> BaseDetector:
    """Instantiate the requested Stage 1 detector backend."""
    key = name.lower()
    if key == "yolo":
        return YoloEllipseDetector(weights=profile.yolo_weights, device=profile.device)
    if key == "sam2":
        return Sam2Detector(yolo_weights=profile.yolo_weights, device=profile.device)
    if key == "sam3":
        return Sam3Detector(device=profile.device)
    raise ValueError(f"Unknown detector backend: {name!r}")


def _draw_detection_overlay(frame: np.ndarray, res: Stage1Result, idx: int) -> np.ndarray:
    """Render ellipse, bbox, crosshair, static-mask tint, and HUD badge on a single frame."""
    vis = frame.copy()
    static_mask = Stage1Result.unpack_static_mask(res.static_masks_packed[idx])
    fg = ~static_mask
    if np.any(fg):
        tint = np.zeros_like(vis)
        tint[fg] = (0, 0, 180)
        vis = cv2.addWeighted(vis, 1.0, tint, 0.15, 0)

    if res.detected[idx]:
        x1, y1, x2, y2 = res.bbox_xyxy[idx].astype(int)
        u, v = res.center_uv[idx]
        a, b = res.semi_axes_ab[idx]
        ang = float(res.ellipse_angle_deg[idx])

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.ellipse(vis, ((u, v), (2.0 * a, 2.0 * b), ang), (0, 255, 0), 2)
        iu, iv = int(round(u)), int(round(v))
        cv2.drawMarker(vis, (iu, iv), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

        flags = [res.det_source[idx].upper()]
        if res.occluded_person[idx]:
            flags.append("OCCLUDED")
        if res.is_blurred[idx]:
            flags.append("BLURRED")
        if res.border_clipped[idx]:
            flags.append("CLIPPED")
        label = f"F{idx} | 2b={2*b:.1f}px | conf={res.conf[idx]:.2f} | {' '.join(flags)}"
        cv2.putText(vis, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    else:
        cv2.putText(vis, f"F{idx} | OUT OF FRAME", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
    return vis


def render_detection_video(video_path: Path, res: Stage1Result, out_path: Path) -> None:
    """Write annotated Stage 1 verification video."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.fps
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CONFIG.camera.width, CONFIG.camera.height))

    for idx in range(len(res.frame_idx)):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(_draw_detection_overlay(frame, res, idx))

    cap.release()
    writer.release()


def save_stage1_diagnostics(video_path: Path, res: Stage1Result, plots_dir: Path) -> None:
    """Save diagnostic keyframe montage and diameter/center curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    keyframes = [0, 110, 158, 165, 185, 204, 205, 250, 381, 385, 420, 461]
    keyframes = [k for k in keyframes if k < len(res.frame_idx)]

    cap = cv2.VideoCapture(str(video_path))
    tiles = []
    for k in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame = cap.read()
        if not ret:
            continue
        vis = _draw_detection_overlay(frame, res, k)
        tile = cv2.resize(vis, (426, 240))
        if res.detected[k]:
            u, v = res.center_uv[k]
            half = int(max(36, 1.4 * res.semi_axes_ab[k, 0]))
            cx, cy = int(round(u)), int(round(v))
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(frame.shape[1], cx + half), min(frame.shape[0], cy + half)
            crop = cv2.resize(vis[y1:y2, x1:x2], (88, 88), interpolation=cv2.INTER_NEAREST)
            tile[-92:-4, -92:-4] = crop
            cv2.rectangle(tile, (-92 + tile.shape[1], -92 + tile.shape[0]), (-4 + tile.shape[1], -4 + tile.shape[0]), (0, 255, 0), 1)
        tiles.append(tile)
    cap.release()

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    montage = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(plots_dir / "stage1_montage.png"), montage)

    frames = res.frame_idx
    det = res.detected
    d_maj = np.where(det, 2.0 * res.semi_axes_ab[:, 0], np.nan)
    d_min = np.where(det, 2.0 * res.semi_axes_ab[:, 1], np.nan)
    eq_d = np.where(det, 2.0 * np.sqrt(np.maximum(res.mask_area_px, 0.0) / np.pi), np.nan)
    ratio = np.where(det, res.semi_axes_ab[:, 0] / np.maximum(res.semi_axes_ab[:, 1], 1e-6), np.nan)
    u_arr = np.where(det, res.center_uv[:, 0], np.nan)
    v_arr = np.where(det, res.center_uv[:, 1], np.nan)

    g0, g1 = CONFIG.video.gap_frames
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax in axes:
        ax.axvspan(g0, g1, color="gray", alpha=0.12)

    axes[0].plot(frames, d_maj, label="2a (major)", lw=1.2, alpha=0.8)
    axes[0].plot(frames, d_min, label="2b (minor)", lw=1.6)
    axes[0].plot(frames, eq_d, "--", label="eq_d (area)", lw=1.2, alpha=0.85)
    axes[0].set_ylabel("Diameter (px)")
    axes[0].set_title("Stage 1: Sub-Pixel Ball Silhouette & Center Trajectory")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(frames, ratio, color="tab:purple", lw=1.3, label="a/b")
    clip_idx = frames[det & res.border_clipped]
    occ_idx = frames[det & res.occluded_person]
    if len(clip_idx):
        axes[1].scatter(clip_idx, ratio[det & res.border_clipped], color="red", s=28, zorder=5, label="border_clipped")
    if len(occ_idx):
        axes[1].scatter(occ_idx, ratio[det & res.occluded_person], color="orange", s=28, zorder=5, label="occluded_person")
    axes[1].axhline(1.25, color="gray", ls=":", lw=1.0, label="blur threshold (1.25)")
    axes[1].set_ylabel("Aspect a/b")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(frames, u_arr, label="u (px)")
    axes[2].plot(frames, v_arr, label="v (px)")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_ylabel("Center (px)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "stage1_diameter_uv.png", dpi=150)
    plt.close(fig)


def run_stage1(profile_name: str = "local", detector_name: str | None = None, max_frames: int | None = None) -> Stage1Result:
    """Execute Stage 1 detection, caching, video overlay, and diagnostic plots."""
    CONFIG.paths.ensure_dirs(profile_name)
    profile = CONFIG.get_profile(profile_name)
    detector = create_detector(detector_name or profile.detector, profile)
    result = detector.run(CONFIG.paths.video, max_frames=max_frames)
    result.save(CONFIG.paths.cache_dir(profile_name))
    render_detection_video(CONFIG.paths.video, result, CONFIG.paths.outputs_dir / "1_detection_2d.mp4")
    save_stage1_diagnostics(CONFIG.paths.video, result, CONFIG.paths.plots_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: 2D football detection & static masking.")
    parser.add_argument("--profile", default="local", choices=("local", "gpu"))
    parser.add_argument("--detector", default=None, choices=("yolo", "sam2", "sam3"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    res = run_stage1(args.profile, args.detector, args.max_frames)
    det = res.detected
    g0, g1 = CONFIG.video.gap_frames
    gap_fps = int(np.sum(det[g0 : g1 + 1])) if len(det) > g1 else 0
    unclipped = det & (~res.border_clipped)
    med_ab = float(np.median(res.semi_axes_ab[unclipped, 0] / res.semi_axes_ab[unclipped, 1])) if np.any(unclipped) else 0.0
    print(f"Detected: {int(det.sum())}/{len(det)} frames | Gap FPs ({g0}-{g1}): {gap_fps}")
    print(f"Median unclipped a/b: {med_ab:.3f} | Blurred: {int(res.is_blurred.sum())} | Occluded: {int(res.occluded_person.sum())} | Clipped: {int(res.border_clipped.sum())}")
    unique, counts = np.unique(res.det_source, return_counts=True)
    print("Sources:", dict(zip(unique.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
