# CPU unit tests for the GPU adapters. Covers SAM mask extraction, PLY and pose parsing, DA3 handoff, and ablation plots.

from __future__ import annotations

from dataclasses import dataclass
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import run_pipeline
import src.detector_2d as det_mod
import src.ego_motion as ego_mod
import src.localizer_3d as loc_mod
import src.tracker_3d as trk_mod
from src.config import CONFIG, PathsConfig, PipelineConfig
from src.detector_2d import Sam2Detector, Sam3Detector, Stage1Result, _extract_ball_record_from_mask
from src.ego_motion import Da3EgoMotion, Stage3Result, _load_da3_camera_poses, _read_ply_xyz


def _dummy_stage1(n_frames: int, ball_center_px: tuple[int, int] | None = None, radius_px: int = 25) -> Stage1Result:
    h, w = CONFIG.camera.height, CONFIG.camera.width
    static = np.ones((h, w), dtype=bool)
    if ball_center_px is not None:
        yy, xx = np.ogrid[:h, :w]
        static[((xx - ball_center_px[0]) ** 2 + (yy - ball_center_px[1]) ** 2) <= radius_px**2] = False
    packed = np.packbits(static.reshape(-1))
    return Stage1Result(
        frame_idx=np.arange(n_frames, dtype=np.int32),
        timestamp_s=np.arange(n_frames, dtype=np.float64) / 30.0,
        detected=np.zeros(n_frames, dtype=bool),
        det_source=np.full(n_frames, "none", dtype=str),
        bbox_xyxy=np.full((n_frames, 4), np.nan, dtype=np.float32),
        center_uv=np.full((n_frames, 2), np.nan, dtype=np.float64),
        semi_axes_ab=np.full((n_frames, 2), np.nan, dtype=np.float64),
        ellipse_angle_deg=np.full(n_frames, np.nan, dtype=np.float64),
        mask_area_px=np.full(n_frames, np.nan, dtype=np.float64),
        conf=np.zeros(n_frames, dtype=np.float32),
        is_blurred=np.zeros(n_frames, dtype=bool),
        occluded_person=np.zeros(n_frames, dtype=bool),
        border_clipped=np.zeros(n_frames, dtype=bool),
        static_masks_packed=np.vstack([packed for _ in range(n_frames)]),
    )


def test_extract_ball_record_from_mask_geometry_and_flags() -> None:
    h, w = 720, 1280
    empty_person = np.zeros((h, w), dtype=bool)

    # 1. Clean circular mask.
    ball_clean = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ball_clean, (640, 360), 22, 1, -1)
    rec, packed = _extract_ball_record_from_mask(0, 0.0, ball_clean > 0, empty_person, 0.92, "sam3", (h, w))
    assert rec.detected and rec.det_source == "sam3"
    assert not rec.is_blurred and not rec.border_clipped and not rec.occluded_person
    assert np.allclose(rec.center_uv, [640.0, 360.0], atol=0.6)
    assert np.allclose(rec.semi_axes_ab, [22.0, 22.0], atol=0.8)
    sm = Stage1Result.unpack_static_mask(packed)
    assert not sm[360, 640]
    assert sm[50, 50]

    # 2. Motion-blurred elongated mask (a/b > 1.25).
    ball_blur = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ball_blur, (500, 420), (32, 16), 25.0, 0, 360, 1, -1)
    rec_blur, _ = _extract_ball_record_from_mask(1, 1 / 30, ball_blur > 0, empty_person, 0.85, "sam2", (h, w))
    assert rec_blur.detected and rec_blur.is_blurred and not rec_blur.border_clipped
    assert abs(float(rec_blur.semi_axes_ab[1]) - 16.0) < 1.0

    # 3. Bottom border-clipped cap + nearby person occlusion.
    ball_clip = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ball_clip, (600, 712), 20, 1, -1)
    person_near = np.zeros((h, w), dtype=bool)
    person_near[690:720, 620:660] = True
    rec_clip, _ = _extract_ball_record_from_mask(2, 2 / 30, ball_clip > 0, person_near, 0.70, "sam2", (h, w))
    assert rec_clip.detected and rec_clip.border_clipped and rec_clip.occluded_person

    # 4. None ball mask produces undetected record with person masked.
    rec_none, packed_none = _extract_ball_record_from_mask(3, 3 / 30, None, person_near, 0.0, "sam3", (h, w))
    assert not rec_none.detected and rec_none.det_source == "none"
    assert not Stage1Result.unpack_static_mask(packed_none)[705, 640]


def test_read_ply_xyz_ascii_and_binary(tmp_path: Path) -> None:
    pts_true = np.array([[1.25, -2.5, 4.0], [0.0, 3.1, 7.25], [-1.5, 0.2, 5.5]], dtype=np.float64)

    ascii_ply = tmp_path / "ascii.ply"
    header_ascii = (
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rows_ascii = "\n".join(f"{p[0]} {p[1]} {p[2]} 120 130 140" for p in pts_true) + "\nnan 0.0 1.0 0 0 0\n"
    ascii_ply.write_text(header_ascii + rows_ascii, encoding="ascii")
    pts_ascii = _read_ply_xyz(ascii_ply)
    assert np.allclose(pts_ascii, pts_true, atol=1e-6)

    bin_ply = tmp_path / "binary.ply"
    header_bin = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    payload = b"".join(struct.pack("<fffBBB", float(p[0]), float(p[1]), float(p[2]), 10, 20, 30) for p in pts_true)
    bin_ply.write_bytes(header_bin + payload)
    pts_bin = _read_ply_xyz(bin_ply)
    assert np.allclose(pts_bin, pts_true, atol=1e-5)


def test_load_da3_camera_poses_all_formats(tmp_path: Path) -> None:
    rots = [Rotation.from_euler("xyz", [5 * i, -8 * i, 12 * i], degrees=True).as_matrix() for i in range(3)]
    ts = [np.array([0.2 * i, -0.1 * i, 1.5], dtype=np.float64) for i in range(3)]

    p16 = tmp_path / "poses16.txt"
    p12 = tmp_path / "poses12.txt"
    ptum = tmp_path / "posestum.txt"

    lines16, lines12, linestum = [], [], []
    for i, (R, t) in enumerate(zip(rots, ts)):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, t
        lines16.append(" ".join(f"{v:.8f}" for v in M.ravel()))
        lines12.append(" ".join(f"{v:.8f}" for v in M[:3, :].ravel()))
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        linestum.append(f"{i/30.0:.4f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}")

    p16.write_text("\n".join(lines16), encoding="utf-8")
    p12.write_text("\n".join(lines12), encoding="utf-8")
    ptum.write_text("\n".join(linestum), encoding="utf-8")

    for path in (p16, p12, ptum):
        R_out, t_out, valid = _load_da3_camera_poses(path, 4)
        assert np.array_equal(valid, [True, True, True, False])
        for i in range(3):
            assert np.allclose(R_out[i], rots[i], atol=1e-5)
            assert np.allclose(t_out[i], ts[i], atol=1e-5)


def test_da3_ego_motion_c2w_and_w2c_auto_detection(tmp_path: Path) -> None:
    n_frames = 6
    K = CONFIG.camera.K

    # Cameras at height Y=-1.65 pitched down 15 deg with 45 deg yaw looking onto floor Y=0.
    R_true = [Rotation.from_euler("yxz", [45.0 + 4.0 * k, -15.0, 0.0], degrees=True).as_matrix() for k in range(n_frames)]
    C_true = [np.array([0.2 * k, -1.65, -0.1 * k], dtype=np.float64) for k in range(n_frames)]

    rng = np.random.default_rng(7)
    fwd0, right0 = R_true[0][:, 2], R_true[0][:, 0]
    ds = rng.uniform(3.5, 7.5, 120)
    xs = rng.uniform(-1.2, 1.2, 120)
    pts_static = np.array([C_true[0] + d * fwd0 + x * right0 for d, x in zip(ds, xs)])
    pts_static[:, 1] = 0.0

    # Dynamic foreground points placed at a fixed world spot and masked per frame at their image projection.
    dyn_center = C_true[0] + 4.0 * R_true[0][:, 2]
    pts_dyn = dyn_center + rng.normal(0.0, 0.02, (25, 3))
    pts_all = np.vstack([pts_static, pts_dyn])

    h, w = CONFIG.camera.height, CONFIG.camera.width
    yy, xx = np.ogrid[:h, :w]
    s1 = _dummy_stage1(n_frames)
    packed_dyn = []
    for k in range(n_frames):
        pc_d = R_true[k].T @ (dyn_center - C_true[k])
        uv_d = (K @ pc_d)[:2] / pc_d[2]
        st = np.ones((h, w), dtype=bool)
        st[((xx - uv_d[0]) ** 2 + (yy - uv_d[1]) ** 2) <= 35**2] = False
        packed_dyn.append(np.packbits(st.reshape(-1)))
    s1.static_masks_packed = np.vstack(packed_dyn)

    for conv in ("c2w", "w2c"):
        cdir = tmp_path / conv
        da3_dir = cdir / "da3"
        (da3_dir / "pcd").mkdir(parents=True, exist_ok=True)

        header = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts_all)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        ).encode("ascii")
        payload = b"".join(struct.pack("<fff", float(p[0]), float(p[1]), float(p[2])) for p in pts_all)
        (da3_dir / "pcd" / "combined_pcd.ply").write_bytes(header + payload)

        lines = []
        for R, C in zip(R_true, C_true):
            M = np.eye(4)
            if conv == "c2w":
                M[:3, :3], M[:3, 3] = R, C
            else:
                M[:3, :3], M[:3, 3] = R.T, -R.T @ C
            lines.append(" ".join(f"{v:.8f}" for v in M.ravel()))
        (da3_dir / "camera_poses.txt").write_text("\n".join(lines), encoding="utf-8")

        R_rec, C_rec, pts_filtered, valid, vo_mode = Da3EgoMotion().estimate(Path("unused.avi"), s1, cdir)
        assert np.all(valid) and np.all(vo_mode == "da3_streaming")
        for k in range(n_frames):
            assert np.allclose(R_rec[k], R_true[k], atol=1e-5)
            assert np.allclose(C_rec[k], C_true[k], atol=1e-5)
        assert len(pts_filtered) < len(pts_all) and len(pts_filtered) >= len(pts_static) - 10


def test_compare_profiles_generates_ablation_plot(tmp_path: Path, monkeypatch) -> None:
    cfg = PipelineConfig(paths=PathsConfig(root=tmp_path))
    cfg.paths.ensure_dirs("local")
    monkeypatch.setattr(run_pipeline, "CONFIG", cfg)

    n = 30
    for prof, noise in (("local", 0.015), ("gpu", 0.008)):
        cdir = cfg.paths.cache_dir(prof)
        s1 = _dummy_stage1(n)
        s1.detected[:] = True
        s1.save(cdir)
        s3 = Stage3Result(
            R_wc=np.tile(np.eye(3), (n, 1, 1)),
            C_w=np.zeros((n, 3)),
            R_gw=np.eye(3),
            t_gw=np.zeros(3),
            R_gc=np.tile(np.eye(3), (n, 1, 1)),
            C_g=np.zeros((n, 3)),
            scale_alpha=0.1 if prof == "local" else 1.0,
            ground_plane_abcd=np.array([0.0, 0.0, 1.0, 0.0]),
            sparse_points_g=np.zeros((0, 3), dtype=np.float32),
            stat_ball_tri_g=np.array([5.0, 0.0, 0.11]),
            pose_valid=np.ones(n, dtype=bool),
            vo_mode=np.full(n, "sfm_kf" if prof == "local" else "da3_streaming", dtype=str),
        )
        s3.save(cdir)
        rng = np.random.default_rng(10 if prof == "local" else 20)
        df4 = pd.DataFrame(
            {
                "frame_idx": np.arange(n, dtype=int),
                "track_status": np.full(n, "observed"),
                "X_g": 5.0 + rng.normal(0.0, noise, n),
                "Y_g": rng.normal(0.0, noise, n),
                "speed_g": np.full(n, 1.2),
                "reproj_err_px": np.abs(rng.normal(0.6 if prof == "local" else 0.4, 0.2, n)),
            }
        )
        df4.to_csv(cdir / "stage4_tracked.csv", index=False)

    run_pipeline._compare_profiles()
    assert (cfg.paths.outputs_dir / "profile_comparison.csv").exists()
    assert (cfg.paths.plots_dir / "ablation_local_vs_gpu.png").exists()


class _TensorStub:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr)

    def __len__(self) -> int:
        return len(self._arr)

    def cpu(self) -> _TensorStub:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _BoxesStub:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls_ids: np.ndarray) -> None:
        self.xyxy = _TensorStub(xyxy)
        self.conf = _TensorStub(conf)
        self.cls = _TensorStub(cls_ids)

    def __len__(self) -> int:
        return len(self._cls_len) if hasattr(self, "_cls_len") else len(self.cls._arr)


class _MasksStub:
    def __init__(self, data: np.ndarray, xy: list[np.ndarray] | None = None) -> None:
        self.data = _TensorStub(data)
        self.xy = xy if xy is not None else []


class _ResultStub:
    def __init__(self, boxes: _BoxesStub | None, masks: _MasksStub | None) -> None:
        self.boxes = boxes
        self.masks = masks


def test_sam2_and_sam3_detector_run_on_real_video_mocked(monkeypatch, tmp_path: Path) -> None:
    s1_real = Stage1Result.load(CONFIG.paths.cache_dir("local"))
    bx1, by1, bx2, by2 = s1_real.bbox_xyxy[0].tolist()
    h, w = CONFIG.camera.height, CONFIG.camera.width
    bcx, bcy = int(round(0.5 * (bx1 + bx2))), int(round(0.5 * (by1 + by2)))

    class FakeYolo:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, img, imgsz=640, classes=None, conf=0.1, device="cpu", verbose=False):
            if img.shape[:2] == (h, w):
                person_poly = np.array([[100, 200], [160, 200], [160, 500], [100, 500]], dtype=np.float32)
                boxes = _BoxesStub(
                    np.array([[100, 200, 160, 500], [bx1, by1, bx2, by2]], dtype=np.float32),
                    np.array([0.90, 0.85], dtype=np.float32),
                    np.array([0, 32], dtype=np.int32),
                )
                return [_ResultStub(boxes, _MasksStub(np.zeros((2, h, w), dtype=np.float32), xy=[person_poly, np.zeros((0, 2))]))]
            return [_ResultStub(None, None)]

    class FakeSam2:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, frame, bboxes=None, device="cpu", verbose=False):
            m = np.zeros((1, h, w), dtype=np.float32)
            cv2.circle(m[0], (bcx, bcy), 16, 1.0, -1)
            return [_ResultStub(None, _MasksStub(m))]

    class FakeSam3Predictor:
        def __init__(self, overrides=None) -> None:
            self.overrides = overrides

        def __call__(self, source=None, text=None, stream=True):
            # Frame 0: ball appears BEFORE person + downscaled tensor + fake skyline/person ball distractors.
            m0 = np.zeros((4, h // 2, w // 2), dtype=np.float32)
            cv2.circle(m0[0], (300, 40), 10, 1.0, -1)       # skyline false positive (cy < 145)
            cv2.circle(m0[1], (60, 160), 10, 1.0, -1)       # inside person mask below
            cv2.circle(m0[2], (bcx // 2, bcy // 2), 9, 1.0, -1)  # true ball
            m0[3, 120:260, 40:80] = 1.0                     # person mask (cid=1 after cid=0)
            yield _ResultStub(
                _BoxesStub(np.zeros((4, 4)), np.array([0.95, 0.92, 0.80, 0.90]), np.array([0, 0, 0, 1])),
                _MasksStub(m0),
            )
            # Frame 1: native resolution ball + person.
            m1 = np.zeros((2, h, w), dtype=np.float32)
            cv2.circle(m1[1], (bcx, bcy), 16, 1.0, -1)
            m1[0, 200:500, 100:160] = 1.0
            yield _ResultStub(
                _BoxesStub(np.zeros((2, 4)), np.array([0.88, 0.84]), np.array([1, 0])),
                _MasksStub(m1),
            )
            # Frame 2: empty gap frame.
            yield _ResultStub(None, None)
            # Frame 3: bottom border-clipped cap.
            m3 = np.zeros((1, h, w), dtype=np.float32)
            cv2.circle(m3[0], (600, h - 6), 18, 1.0, -1)
            yield _ResultStub(_BoxesStub(np.zeros((1, 4)), np.array([0.75]), np.array([0])), _MasksStub(m3))

    monkeypatch.setattr(det_mod, "YOLO", FakeYolo)
    monkeypatch.setattr("ultralytics.SAM", FakeSam2)
    monkeypatch.setattr("ultralytics.models.sam.SAM3VideoSemanticPredictor", FakeSam3Predictor)

    res2 = Sam2Detector(device="cpu").run(CONFIG.paths.video, max_frames=4)
    assert len(res2.frame_idx) == 4 and int(res2.detected.sum()) == 4
    assert np.all(res2.det_source == "sam2")

    fake_pt = tmp_path / "sam3.pt"
    fake_pt.write_bytes(b"stub")
    res3 = Sam3Detector(weights=str(fake_pt), device="cpu").run(CONFIG.paths.video, max_frames=4)
    assert np.array_equal(res3.detected, [True, True, False, True])
    assert res3.det_source[0] == "sam3" and res3.det_source[2] == "none"
    assert abs(float(res3.center_uv[0, 0]) - bcx) < 3.0 and bool(res3.border_clipped[3])


@dataclass(frozen=True)
class _TempPaths(PathsConfig):
    source_video: Path = CONFIG.paths.video

    @property
    def video(self) -> Path:
        return self.source_video


def test_export_frames_and_gpu_da3_end_to_end_handoff(tmp_path: Path, monkeypatch) -> None:
    cfg = PipelineConfig(paths=_TempPaths(root=tmp_path, source_video=CONFIG.paths.video))
    cfg.paths.ensure_dirs("gpu")
    for mod in (run_pipeline, ego_mod, loc_mod, trk_mod):
        monkeypatch.setattr(mod, "CONFIG", cfg)

    frames_dir = run_pipeline._export_video_frames("gpu", max_frames=6)
    assert len(list(frames_dir.glob("frame_*.jpg"))) == 6

    real_local = CONFIG.paths.cache_dir("local")
    for fname in ("stage1_2d.npz", "stage2_3d.csv", "stage3_ego.npz", "stage4_tracked.csv"):
        shutil.copy2(real_local / fname, cfg.paths.cache_dir("local") / fname)
    for fname in ("stage1_2d.npz", "stage2_3d.csv"):
        shutil.copy2(real_local / fname, cfg.paths.cache_dir("gpu") / fname)

    s3_real = Stage3Result.load(real_local)
    scale_syn = 11.3
    R_rand = Rotation.from_euler("xyz", [28.0, -35.0, 14.0], degrees=True).as_matrix()
    t_rand = np.array([1.2, -0.7, 2.5], dtype=np.float64)

    da3_dir = cfg.paths.cache_dir("gpu") / "da3"
    (da3_dir / "pcd").mkdir(parents=True, exist_ok=True)
    pts_syn = scale_syn * ((R_rand @ s3_real.sparse_points_g.astype(np.float64).T).T + t_rand)
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts_syn)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    payload = b"".join(struct.pack("<fff", float(p[0]), float(p[1]), float(p[2])) for p in pts_syn)
    (da3_dir / "pcd" / "combined_pcd.ply").write_bytes(header + payload)

    pose_lines = []
    for k in range(len(s3_real.pose_valid)):
        M = np.eye(4)
        M[:3, :3] = R_rand @ s3_real.R_gc[k]
        M[:3, 3] = scale_syn * (R_rand @ s3_real.C_g[k] + t_rand)
        pose_lines.append(" ".join(f"{v:.8f}" for v in M.ravel()))
    (da3_dir / "camera_poses.txt").write_text("\n".join(pose_lines), encoding="utf-8")

    s3_gpu = ego_mod.run_stage3("gpu", vo_name="da3")
    assert int(s3_gpu.pose_valid.sum()) == 462
    assert abs(s3_gpu.scale_alpha - (1.0 / scale_syn)) < 0.008
    assert 1.55 <= float(np.median(s3_gpu.C_g[:, 2])) <= 1.80

    df4_gpu, _ = trk_mod.run_stage4("gpu")
    s0, s1_end = cfg.video.stationary_frames
    stat_ok = df4_gpu.iloc[s0 : s1_end + 1]
    stat_ok = stat_ok[stat_ok["track_status"] != "gap"]
    stat_std_cm = stat_ok[["X_g", "Y_g"]].to_numpy().std(axis=0) * 100.0
    assert float(stat_std_cm[0]) < 1.0 and float(stat_std_cm[1]) < 1.0
    assert sorted(int(s) for s in df4_gpu["segment_id"].unique() if int(s) >= 0) == [0, 1, 2]

    run_pipeline._compare_profiles()
    assert (cfg.paths.outputs_dir / "profile_comparison.csv").exists()
    assert (cfg.paths.plots_dir / "ablation_local_vs_gpu.png").exists()


def test_cloud_script_and_requirements_gpu_hygiene() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_sh = (root / "scripts" / "run_cloud_gpu.sh").read_bytes()
    assert b"\r\n" not in raw_sh
    txt_sh = raw_sh.decode("utf-8")
    assert "da3_streaming/da3_streaming.py" in txt_sh and "nvidia-smi" in txt_sh

    req_gpu = (root / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "sam2>=" not in req_gpu and "ultralytics>=" in req_gpu
