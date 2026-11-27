# Main CLI entry point. Runs pipeline stages 1 through 5, exports video frames, and compares local vs gpu profiles.

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG


def _parse_stages(stage_arg: str) -> list[int]:
    val_str = stage_arg.strip().lower()
    if val_str == "none":
        return []
    if val_str == "all":
        return [1, 2, 3, 4, 5]
    stages: list[int] = []
    for tok in stage_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if val not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid stage {val}; expected 1..5, 'all', or 'none'.")
        if val not in stages:
            stages.append(val)
    if not stages:
        raise ValueError("No stages selected.")
    return stages


def _export_video_frames(profile_name: str, max_frames: int | None = None) -> Path:
    import cv2

    out_dir = CONFIG.paths.cache_dir(profile_name) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(CONFIG.paths.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {CONFIG.paths.video}")
    idx = 0
    while max_frames is None or idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    print(f"Exported {idx} frames -> {out_dir}")
    return out_dir


def _save_ablation_plot(data_by_prof: dict[str, tuple], df_cmp: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    df_l, errs_l = data_by_prof["local"]
    df_g, errs_g = data_by_prof["gpu"]

    # Top-left: Pre-kick stationary XY scatter std comparison.
    row_l = df_cmp[df_cmp["profile"] == "local"].iloc[0]
    row_g = df_cmp[df_cmp["profile"] == "gpu"].iloc[0]
    x_pos = np.arange(2)
    axes[0, 0].bar(x_pos - 0.16, [row_l["prekick_std_x_cm"], row_l["prekick_std_y_cm"]], width=0.32, label="local")
    axes[0, 0].bar(x_pos + 0.16, [row_g["prekick_std_x_cm"], row_g["prekick_std_y_cm"]], width=0.32, label="gpu")
    axes[0, 0].set_xticks(x_pos, ["X_g Std (cm)", "Y_g Std (cm)"])
    axes[0, 0].set_ylabel("Pre-Kick Stationary Std (cm)")
    axes[0, 0].set_title("Stationary Ball Stability (Lower Better)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Top-right: Unclipped reprojection error distribution.
    axes[0, 1].boxplot([errs_l, errs_g], tick_labels=["local", "gpu"], showfliers=False)
    axes[0, 1].set_ylabel("Unclipped Reproj Error (px)")
    axes[0, 1].set_title("2D Reprojection Error Distribution")
    axes[0, 1].grid(alpha=0.3)

    # Bottom-left: Ground-frame forward trajectory comparison.
    axes[1, 0].plot(df_l["frame_idx"], df_l["X_g"], lw=1.8, label="local X_g")
    axes[1, 0].plot(df_g["frame_idx"], df_g["X_g"], "--", lw=1.6, label="gpu X_g")
    axes[1, 0].set_xlabel("Frame Index")
    axes[1, 0].set_ylabel("Ground Forward X_g (m)")
    axes[1, 0].set_title("Trajectory Agreement")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Bottom-right: Ground-frame speed comparison.
    axes[1, 1].plot(df_l["frame_idx"], df_l["speed_g"], lw=1.8, label="local |v_g|")
    axes[1, 1].plot(df_g["frame_idx"], df_g["speed_g"], "--", lw=1.6, label="gpu |v_g|")
    axes[1, 1].set_xlabel("Frame Index")
    axes[1, 1].set_ylabel("World Speed |v_g| (m/s)")
    axes[1, 1].set_title("Velocity Profile Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compare_profiles() -> None:
    local_dir = CONFIG.paths.cache_dir("local")
    gpu_dir = CONFIG.paths.cache_dir("gpu")
    req_files = ["stage1_2d.npz", "stage3_ego.npz", "stage4_tracked.csv"]
    missing = [f for f in req_files if not (local_dir / f).exists() or not (gpu_dir / f).exists()]
    if missing:
        print(
            "Profile comparison skipped: both 'cache/local' and 'cache/gpu' must be populated first "
            f"(missing: {missing})."
        )
        return

    from src.detector_2d import Stage1Result
    from src.ego_motion import Stage3Result

    rows = []
    data_by_prof = {}
    for prof in ("local", "gpu"):
        cdir = CONFIG.paths.cache_dir(prof)
        s1 = Stage1Result.load(cdir)
        s3 = Stage3Result.load(cdir)
        df4 = pd.read_csv(cdir / "stage4_tracked.csv")
        g0, g1 = CONFIG.video.gap_frames
        s0, s1_end = CONFIG.video.stationary_frames
        n_stat_end = min(s1_end + 1, len(df4))
        stat_ok = df4.iloc[s0:n_stat_end]
        stat_ok = stat_ok[stat_ok["track_status"] != "gap"]
        stat_std = stat_ok[["X_g", "Y_g"]].to_numpy().std(axis=0) * 100.0 if len(stat_ok) else np.zeros(2)
        errs = df4["reproj_err_px"].to_numpy(dtype=np.float64)
        errs_u = errs[np.isfinite(errs) & (~s1.border_clipped[: len(errs)])]
        data_by_prof[prof] = (df4, errs_u)
        rows.append(
            {
                "profile": prof,
                "detected_frames": int(s1.detected.sum()),
                "gap_fp": int(s1.detected[g0 : min(g1 + 1, len(s1.detected))].sum()) if len(s1.detected) > g0 else 0,
                "pose_valid_frames": int(s3.pose_valid.sum()),
                "scale_alpha": float(s3.scale_alpha),
                "prekick_std_x_cm": float(stat_std[0]),
                "prekick_std_y_cm": float(stat_std[1]),
                "unclipped_reproj_med_px": float(np.median(errs_u)) if len(errs_u) else float("nan"),
                "unclipped_reproj_p95_px": float(np.percentile(errs_u, 95)) if len(errs_u) else float("nan"),
            }
        )
    df_cmp = pd.DataFrame(rows)
    CONFIG.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG.paths.outputs_dir / "profile_comparison.csv"
    df_cmp.to_csv(out_path, index=False)
    plot_path = CONFIG.paths.plots_dir / "ablation_local_vs_gpu.png"
    _save_ablation_plot(data_by_prof, df_cmp, plot_path)
    print(f"\nProfile comparison written to {out_path} and {plot_path}:")
    print(df_cmp.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="dotLumen Computer Vision Challenge 3 Pipeline")
    parser.add_argument("--profile", default="local", choices=("local", "gpu"), help="Execution profile.")
    parser.add_argument("--stage", default="all", help="Stage number(s): 1..5, comma-separated, 'all', or 'none'.")
    parser.add_argument("--detector", default=None, choices=("yolo", "sam2", "sam3"), help="Override Stage 1 detector.")
    parser.add_argument("--vo", default=None, choices=("pycolmap", "sift", "da3", "megasam"), help="Override Stage 3 VO.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for Stage 1 smoke runs.")
    parser.add_argument("--pred-steps", type=int, default=24, help="Forward prediction steps in Stage 4.")
    parser.add_argument("--export-frames", action="store_true", help="Export RGB JPEGs to cache/<profile>/frames/.")
    parser.add_argument("--compare-profiles", action="store_true", help="Compare cached local vs. gpu metrics.")
    args = parser.parse_args()

    stages = _parse_stages(args.stage)
    CONFIG.paths.ensure_dirs(args.profile)
    print(f"Running profile={args.profile!r}, stages={stages}")
    t_total = time.perf_counter()

    if args.export_frames:
        _export_video_frames(args.profile, max_frames=args.max_frames)

    if 1 in stages:
        from src.detector_2d import run_stage1

        t0 = time.perf_counter()
        s1 = run_stage1(profile_name=args.profile, detector_name=args.detector, max_frames=args.max_frames)
        print(f"[Stage 1] Done in {time.perf_counter() - t0:.1f}s | detected={int(s1.detected.sum())}/{len(s1.detected)}")

    if 2 in stages:
        from src.localizer_3d import run_stage2

        t0 = time.perf_counter()
        df2 = run_stage2(profile_name=args.profile)
        print(f"[Stage 2] Done in {time.perf_counter() - t0:.1f}s | valid={int(df2['valid'].sum())}/{len(df2)}")

    if 3 in stages:
        from src.ego_motion import run_stage3

        t0 = time.perf_counter()
        s3 = run_stage3(profile_name=args.profile, vo_name=args.vo)
        print(f"[Stage 3] Done in {time.perf_counter() - t0:.1f}s | poses={int(s3.pose_valid.sum())}/{len(s3.pose_valid)}, alpha={s3.scale_alpha:.5f}")

    if 4 in stages:
        from src.tracker_3d import run_stage4

        t0 = time.perf_counter()
        df4, _ = run_stage4(profile_name=args.profile, pred_steps=args.pred_steps)
        s0, s1_end = CONFIG.video.stationary_frames
        stat_ok = df4.iloc[s0 : s1_end + 1]
        stat_ok = stat_ok[stat_ok["track_status"] != "gap"]
        stat_std = stat_ok[["X_g", "Y_g"]].to_numpy().std(axis=0) * 100.0
        print(f"[Stage 4] Done in {time.perf_counter() - t0:.1f}s | pre-kick XY std={stat_std.round(2)} cm")

    if 5 in stages:
        from src.visualizer import run_stage5

        t0 = time.perf_counter()
        out_paths = run_stage5(profile_name=args.profile)
        print(f"[Stage 5] Done in {time.perf_counter() - t0:.1f}s")
        for key, path in out_paths.items():
            print(f"  {key}: {Path(path)}")

    if args.compare_profiles:
        _compare_profiles()

    print(f"Total elapsed: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
