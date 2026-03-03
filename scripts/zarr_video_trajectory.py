#!/usr/bin/env python3
"""Export a synchronized video: RGB frames + end-effector trajectory panels.

Left panel: camera frame (2D video)
Right-top: 3D end-effector trajectory trail
Right-bottom: XY projection trail

Notes:
- This is time-synchronized visualization, not pixel-accurate projection.
- Pixel-accurate overlay on the camera frame requires camera calibration.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "pick_cube.zarr"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "rgb_with_trajectory.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synchronized RGB + trajectory video from UMI-format Zarr."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR_PATH,
        help="Path to UMI Zarr store.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="camera0_rgb",
        help="Array key under /data for RGB frames.",
    )
    parser.add_argument(
        "--eef-key",
        type=str,
        default="robot0_eef_pos",
        help="Array key under /data for end-effector position.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--trail",
        type=int,
        default=120,
        help="Number of past points to show in trajectory trail.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1 for all frames).",
    )

    # Absolute index mode
    parser.add_argument("--start", type=int, default=0, help="Absolute start index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Absolute end index (exclusive).")

    # Episode mode (overrides absolute start/end if provided)
    parser.add_argument("--episode", type=int, default=None, help="Episode index to visualize.")
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="Relative start in selected episode (inclusive).",
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=None,
        help="Relative end in selected episode (exclusive).",
    )
    return parser.parse_args()


def get_episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    if episode_ends.ndim != 1:
        raise ValueError("/meta/episode_ends must be 1D.")
    if len(episode_ends) == 0:
        return []
    if np.any(np.diff(episode_ends) <= 0):
        raise ValueError("/meta/episode_ends must be strictly increasing.")
    starts = np.concatenate(([0], episode_ends[:-1]))
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def resolve_range(args: argparse.Namespace, total_steps: int, episode_bounds: list[tuple[int, int]]) -> tuple[int, int]:
    if args.episode is None:
        start = args.start
        end = total_steps if args.end is None else args.end
    else:
        if args.episode < 0 or args.episode >= len(episode_bounds):
            raise ValueError(f"--episode out of range [0, {len(episode_bounds)-1}]")
        ep_start_abs, ep_end_abs = episode_bounds[args.episode]
        ep_len = ep_end_abs - ep_start_abs
        rel_start = args.episode_start
        rel_end = ep_len if args.episode_end is None else args.episode_end
        if rel_start < 0 or rel_end > ep_len or rel_start >= rel_end:
            raise ValueError(
                f"Invalid episode slice [{rel_start}, {rel_end}) for episode length {ep_len}."
            )
        start = ep_start_abs + rel_start
        end = ep_start_abs + rel_end

    if start < 0 or end > total_steps or start >= end:
        raise ValueError(f"Invalid absolute range [{start}, {end}) with total steps {total_steps}.")
    return start, end


def render_video(
    rgb_arr,
    eef_pos: np.ndarray,
    start: int,
    end: int,
    stride: int,
    trail: int,
    fps: int,
    output_path: Path,
) -> None:
    selected = np.arange(start, end, stride, dtype=np.int64)
    if len(selected) == 0:
        raise ValueError("No frames selected. Check start/end/stride.")

    segment = eef_pos[start:end]
    mins = segment.min(axis=0)
    maxs = segment.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.10 * span
    xyz_min = mins - pad
    xyz_max = maxs + pad

    first_frame = np.asarray(rgb_arr[int(selected[0])])
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(
            f"Expected RGB shape [H, W, 3], got {first_frame.shape} for index {int(selected[0])}."
        )

    fig = plt.figure(figsize=(13, 6), dpi=120)
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_xy = fig.add_subplot(gs[1, 1])

    img_artist = ax_img.imshow(first_frame)
    ax_img.set_title("Camera Frame (2D)")
    ax_img.axis("off")

    (line3d,) = ax_3d.plot([], [], [], linewidth=1.8, color="tab:blue")
    point3d = ax_3d.scatter([], [], [], color="red", s=22)
    ax_3d.set_xlim(xyz_min[0], xyz_max[0])
    ax_3d.set_ylim(xyz_min[1], xyz_max[1])
    ax_3d.set_zlim(xyz_min[2], xyz_max[2])
    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")
    ax_3d.set_title("EEF 3D Trajectory")

    (line_xy,) = ax_xy.plot([], [], linewidth=1.8, color="tab:orange")
    (point_xy,) = ax_xy.plot([], [], marker="o", markersize=6, color="red")
    ax_xy.set_xlim(xyz_min[0], xyz_max[0])
    ax_xy.set_ylim(xyz_min[1], xyz_max[1])
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("EEF XY Projection")
    ax_xy.grid(True, alpha=0.3)
    status_text = ax_xy.text(0.02, 0.95, "", transform=ax_xy.transAxes, va="top")

    fig.tight_layout()
    fig.canvas.draw()
    canvas_w, canvas_h = fig.canvas.get_width_height()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open cv2.VideoWriter for output MP4.")

    try:
        for i, abs_idx in enumerate(selected):
            frame = np.asarray(rgb_arr[int(abs_idx)])
            img_artist.set_data(frame)

            rel_idx = int(abs_idx - start)
            tail_start = max(0, rel_idx - trail + 1)
            trail_xyz = segment[tail_start : rel_idx + 1]

            x = trail_xyz[:, 0]
            y = trail_xyz[:, 1]
            z = trail_xyz[:, 2]

            line3d.set_data(x, y)
            line3d.set_3d_properties(z)
            point3d._offsets3d = ([x[-1]], [y[-1]], [z[-1]])

            line_xy.set_data(x, y)
            point_xy.set_data([x[-1]], [y[-1]])
            status_text.set_text(f"t={int(abs_idx)}  frame={i+1}/{len(selected)}")

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")
    if args.fps <= 0:
        raise ValueError("--fps must be >= 1")
    if args.trail <= 0:
        raise ValueError("--trail must be >= 1")

    if not args.zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {args.zarr_path}")

    warnings.filterwarnings(
        "ignore",
        message=r"Object at \.DS_Store is not recognized as a component of a Zarr hierarchy\.",
    )
    root = zarr.open(str(args.zarr_path), mode="r")

    if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError("Expected UMI layout: /data and /meta/episode_ends")
    if args.camera_key not in root["data"]:
        raise KeyError(f"Missing /data/{args.camera_key}")
    if args.eef_key not in root["data"]:
        raise KeyError(f"Missing /data/{args.eef_key}")

    rgb_arr = root["data"][args.camera_key]
    eef_pos = np.asarray(root["data"][args.eef_key], dtype=np.float32)
    if eef_pos.ndim != 2 or eef_pos.shape[1] < 3:
        raise ValueError(f"Expected eef array shape [T, 3+], got {eef_pos.shape}")

    total_steps = min(int(rgb_arr.shape[0]), int(eef_pos.shape[0]))
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    episode_bounds = get_episode_bounds(episode_ends)

    start, end = resolve_range(args, total_steps, episode_bounds)
    print(f"Creating video from [{start}, {end}) with stride={args.stride}, fps={args.fps}, trail={args.trail}")
    print(f"Output: {args.output.resolve()}")
    if args.episode is not None:
        print(f"Episode mode: episode={args.episode}")
    print(
        "Note: left panel is 2D camera image; trajectory is shown in robot coordinates "
        "(synchronized by timestep, not projected into pixels)."
    )

    render_video(
        rgb_arr=rgb_arr,
        eef_pos=eef_pos[:, :3],
        start=start,
        end=end,
        stride=args.stride,
        trail=args.trail,
        fps=args.fps,
        output_path=args.output,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
