#!/usr/bin/env python3
"""Render full URDF kinematic motion and run approximate self-collision checks.

Outputs:
- Animated GIF of full robot motion (URDF kinematic chain skeleton).
- JSON summary report with collision statistics.
- CSV of per-frame collision flags.

Collision check method:
- Approximate each link by a capsule centered on its kinematic segment.
- Collision if shortest segment-segment distance < (r_i + r_j) for non-adjacent links.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from ikpy.chain import Chain
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render URDF motion and approximate self-collision report."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/datasets/lerobot_umi_pick_cube_so101_v3"),
        help="LeRobot dataset root.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/assets/urdf/so101_new_calib.urdf"),
        help="URDF path.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index.")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame.")
    parser.add_argument("--fps", type=int, default=20, help="GIF fps.")
    parser.add_argument("--trail", type=int, default=50, help="TCP trail length.")
    parser.add_argument(
        "--link-radius",
        type=float,
        default=0.025,
        help="Approx capsule radius for each link (m).",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/videos/so101_urdf_motion_collision_ep0.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/reports/so101_collision_ep0.json"),
        help="Output JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/reports/so101_collision_ep0_frames.csv"),
        help="Output CSV report.",
    )
    return parser.parse_args()


def load_actions(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = json.loads(info_path.read_text())
    action_ft = info.get("features", {}).get("action")
    if action_ft is None:
        raise KeyError("action feature missing in info.json")
    action_names = action_ft.get("names", [f"joint_{i+1}" for i in range(6)])

    table = pq.ParquetDataset(str(dataset_root / "data")).read(columns=["action", "episode_index"])
    d = table.to_pydict()
    action = np.asarray(d["action"], dtype=np.float64)
    ep = np.asarray(d["episode_index"], dtype=np.int64)
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"Expected action shape [N, >=6], got {action.shape}")
    return action[:, :6], ep, action_names[:6]


def build_chain(urdf: Path) -> tuple[Chain, list[int], list[str]]:
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    warnings.filterwarnings("ignore", category=UserWarning, module="ikpy")
    chain = Chain.from_urdf_file(str(urdf))
    movable_indices = [
        i for i, link in enumerate(chain.links) if getattr(link, "joint_type", "fixed") != "fixed"
    ]
    if len(movable_indices) < 5:
        raise ValueError(f"Need at least 5 movable joints in URDF, got {len(movable_indices)}")
    return chain, movable_indices[:5], [link.name for link in chain.links]


def frame_transforms(chain: Chain, movable_indices: list[int], q5: np.ndarray) -> np.ndarray:
    """Return [L,4,4] transform for each chain link."""
    q_full = np.zeros((len(chain.links),), dtype=np.float64)
    for j, idx in enumerate(movable_indices):
        q_full[idx] = q5[j]
    mats = chain.forward_kinematics(q_full, full_kinematics=True)
    return np.asarray(mats, dtype=np.float64)


def segment_segment_distance(a0, a1, b0, b1) -> float:
    """Closest distance between two 3D segments."""
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    den = a * c - b * b
    eps = 1e-12

    if den < eps:
        s = 0.0
        t = np.clip(e / (c + eps), 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / den, 0.0, 1.0)
        t = np.clip((a * e - b * d) / den, 0.0, 1.0)

    p = a0 + s * u
    q = b0 + t * v
    return float(np.linalg.norm(p - q))


def compute_collision_stats(
    link_points: np.ndarray,
    link_radius: float,
    link_names: list[str],
) -> tuple[list[dict], dict]:
    """
    link_points shape: [T, L, 3], where consecutive points form segments.
    """
    t_count, l_count, _ = link_points.shape
    seg_count = l_count - 1

    # Non-adjacent segment pairs only
    seg_pairs = []
    for i in range(seg_count):
        for j in range(i + 1, seg_count):
            if abs(i - j) <= 1:
                continue
            seg_pairs.append((i, j))

    frame_rows = []
    pair_counter = Counter()
    overall_min_dist = math.inf
    overall_min_pair = None
    min_margin = math.inf

    for t in range(t_count):
        pts = link_points[t]
        collided_pairs = []
        min_dist_frame = math.inf
        min_pair_frame = None

        for i, j in seg_pairs:
            d = segment_segment_distance(pts[i], pts[i + 1], pts[j], pts[j + 1])
            if d < min_dist_frame:
                min_dist_frame = d
                min_pair_frame = (i, j)
            if d < overall_min_dist:
                overall_min_dist = d
                overall_min_pair = (i, j)

            margin = d - 2.0 * link_radius
            if margin < min_margin:
                min_margin = margin

            if d < 2.0 * link_radius:
                collided_pairs.append((i, j))
                pair_counter[(i, j)] += 1

        frame_rows.append(
            {
                "frame": t,
                "collision": int(len(collided_pairs) > 0),
                "num_pairs": len(collided_pairs),
                "pairs": collided_pairs,
                "closest_pair": min_pair_frame,
                "closest_dist_m": float(min_dist_frame),
            }
        )

    collided_frames = sum(r["collision"] for r in frame_rows)
    summary = {
        "frames_total": t_count,
        "frames_colliding": int(collided_frames),
        "collision_rate": float(collided_frames / max(t_count, 1)),
        "link_radius_m": float(link_radius),
        "overall_min_segment_distance_m": float(overall_min_dist),
        "overall_min_margin_m": float(min_margin),
        "overall_min_pair_segments": overall_min_pair,
        "overall_min_pair_links": None
        if overall_min_pair is None
        else [link_names[overall_min_pair[0]], link_names[overall_min_pair[1]]],
        "top_collision_pairs": [
            {
                "pair_segments": [i, j],
                "pair_links": [link_names[i], link_names[j]],
                "frames": int(count),
            }
            for (i, j), count in pair_counter.most_common(10)
        ],
    }
    return frame_rows, summary


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half = max(float(np.max(maxs - mins) / 2.0), 1e-3)
    half *= 1.25
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render_gif(
    link_points: np.ndarray,
    actions: np.ndarray,
    action_names: list[str],
    frame_rows: list[dict],
    output_gif: Path,
    fps: int,
    trail: int,
) -> None:
    t_count, l_count, _ = link_points.shape
    seg_count = l_count - 1
    tcp = link_points[:, -1, :]

    fig = plt.figure(figsize=(12, 6), dpi=120)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axa = fig.add_subplot(1, 2, 2)

    # 3D artists
    seg_lines = []
    for _ in range(seg_count):
        (ln,) = ax3d.plot([], [], [], lw=3.0, color="tab:blue")
        seg_lines.append(ln)
    (joints_scatter,) = ax3d.plot([], [], [], "o", color="k", markersize=4)
    (trail_line,) = ax3d.plot([], [], [], lw=1.5, color="tab:gray", alpha=0.8)
    text3d = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes, va="top")

    set_axes_equal(ax3d, link_points.reshape(-1, 3))
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("SO-101 URDF Kinematic Motion")

    # 6DoF traces
    x = np.arange(t_count)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for j in range(6):
        axa.plot(x, actions[:, j], lw=1.0, color=colors[j], label=action_names[j])
    vline = axa.axvline(0, color="k", lw=1.2)
    axa.set_title("Retargeted 6DoF action")
    axa.set_xlabel("frame")
    axa.grid(alpha=0.3)
    axa.legend(loc="upper right", fontsize=8)

    def update(i: int):
        pts = link_points[i]
        row = frame_rows[i]
        pair_set = set(tuple(p) for p in row["pairs"])
        # color segments red if involved in any collision pair
        colliding_segments = set()
        for a, b in pair_set:
            colliding_segments.add(a)
            colliding_segments.add(b)

        for s in range(seg_count):
            p0 = pts[s]
            p1 = pts[s + 1]
            seg_lines[s].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            seg_lines[s].set_3d_properties([p0[2], p1[2]])
            seg_lines[s].set_color("crimson" if s in colliding_segments else "tab:blue")

        joints_scatter.set_data(pts[:, 0], pts[:, 1])
        joints_scatter.set_3d_properties(pts[:, 2])

        t0 = max(0, i - trail + 1)
        tr = tcp[t0 : i + 1]
        trail_line.set_data(tr[:, 0], tr[:, 1])
        trail_line.set_3d_properties(tr[:, 2])

        if row["collision"]:
            status = f"COLLISION (approx) pairs={row['num_pairs']}"
        else:
            status = "No collision (approx)"
        text3d.set_text(
            f"frame={i+1}/{t_count} | {status}\n"
            f"closest_dist={row['closest_dist_m']:.4f} m"
        )

        vline.set_xdata([i, i])
        return [*seg_lines, joints_scatter, trail_line, text3d, vline]

    anim = FuncAnimation(fig, update, frames=t_count, interval=1000 / max(fps, 1), blit=False)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_gif), writer=PillowWriter(fps=fps))
    plt.close(fig)


def write_reports(frame_rows: list[dict], summary: dict, out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "collision", "num_pairs", "closest_dist_m", "pairs"])
        for r in frame_rows:
            writer.writerow(
                [
                    r["frame"],
                    r["collision"],
                    r["num_pairs"],
                    f"{r['closest_dist_m']:.6f}",
                    ";".join(f"{a}-{b}" for a, b in r["pairs"]),
                ]
            )


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.trail < 1:
        raise ValueError("--trail must be >= 1")
    if args.link_radius <= 0:
        raise ValueError("--link-radius must be > 0")

    actions, ep_idx, action_names = load_actions(args.dataset_root)
    valid_eps = sorted(np.unique(ep_idx).tolist())
    if args.episode not in valid_eps:
        raise ValueError(f"Episode {args.episode} not in dataset. Available: {valid_eps}")

    actions_ep = actions[ep_idx == args.episode][:: args.stride]
    if len(actions_ep) < 2:
        raise ValueError("Episode too short after applying stride.")

    chain, movable, link_names = build_chain(args.urdf)
    # use first 5 action channels as arm joints
    q5 = actions_ep[:, :5]

    # Compute all link points over time.
    link_points = np.zeros((len(q5), len(chain.links), 3), dtype=np.float64)
    for i, q in enumerate(q5):
        mats = frame_transforms(chain, movable, q)
        link_points[i] = mats[:, :3, 3]

    frame_rows, summary = compute_collision_stats(
        link_points=link_points,
        link_radius=args.link_radius,
        link_names=link_names,
    )

    print(
        f"Episode={args.episode} frames={len(actions_ep)} stride={args.stride} "
        f"collision_rate={summary['collision_rate']:.3f}"
    )
    if summary["top_collision_pairs"]:
        print("Top collision pairs (approx):")
        for row in summary["top_collision_pairs"][:5]:
            print(" ", row)

    render_gif(
        link_points=link_points,
        actions=actions_ep,
        action_names=action_names,
        frame_rows=frame_rows,
        output_gif=args.output_gif,
        fps=args.fps,
        trail=args.trail,
    )
    write_reports(frame_rows, summary, args.output_json, args.output_csv)
    print(f"Wrote GIF: {args.output_gif}")
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
