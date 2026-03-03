#!/usr/bin/env python3
"""Render side-by-side debug GIF: true video frame + URDF kinematic motion.

Uses LeRobotDataset decoded frames (video_backend='pyav') for the left panel and
IKPy forward kinematics from action[0:5] for the right panel.
Also runs approximate self-collision checks (capsule distance on link segments).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    parser = argparse.ArgumentParser(description="Render camera+URDF side-by-side debug GIF.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/datasets/lerobot_umi_pick_cube_so101_v3"),
        help="LeRobot dataset root.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/umi_pick_cube_so101_v3",
        help="LeRobot repo_id used for local dataset loading.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/assets/urdf/so101_new_calib.urdf"),
        help="URDF path.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index.")
    parser.add_argument("--stride", type=int, default=2, help="Use every N-th frame.")
    parser.add_argument("--fps", type=int, default=18, help="Output GIF fps.")
    parser.add_argument("--trail", type=int, default=50, help="TCP trail length.")
    parser.add_argument(
        "--link-radius",
        type=float,
        default=0.025,
        help="Approx capsule radius for each link (meters).",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/videos/so101_video_urdf_debug_ep0.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/reports/so101_video_urdf_debug_ep0.json"),
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/reports/so101_video_urdf_debug_ep0_frames.csv"),
        help="Output per-frame CSV.",
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=Path("/Users/rebnoob/lerobot/src"),
        help="Path to lerobot/src for import bootstrap.",
    )
    return parser.parse_args()


def bootstrap_lerobot(lerobot_src: Path):
    if str(lerobot_src) not in sys.path and lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError(
            f"Failed to import lerobot. Tried path: {lerobot_src}\n"
            f"Install with: pip install -e /Users/rebnoob/lerobot"
        ) from exc
    return LeRobotDataset


def load_metadata(dataset_root: Path) -> tuple[np.ndarray, list[str], str]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = json.loads(info_path.read_text())

    camera_keys = [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    if not camera_keys:
        raise KeyError("No video feature found in dataset info.json")
    camera_key = camera_keys[0]

    action_ft = info.get("features", {}).get("action")
    if action_ft is None:
        raise KeyError("action feature missing in info.json")
    action_names = action_ft.get("names", [f"joint_{i+1}" for i in range(6)])

    table = pq.ParquetDataset(str(dataset_root / "data")).read(columns=["episode_index", "action"])
    d = table.to_pydict()
    ep_idx = np.asarray(d["episode_index"], dtype=np.int64)
    action = np.asarray(d["action"], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"Expected action shape [N, >=6], got {action.shape}")
    return ep_idx, action[:, :6], action_names[:6], camera_key


def build_chain(urdf: Path) -> tuple[Chain, list[int], list[str]]:
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    warnings.filterwarnings("ignore", category=UserWarning, module="ikpy")
    chain = Chain.from_urdf_file(str(urdf))
    movable = [i for i, l in enumerate(chain.links) if getattr(l, "joint_type", "fixed") != "fixed"]
    if len(movable) < 5:
        raise ValueError(f"Need >=5 movable joints in URDF, got {len(movable)}")
    return chain, movable[:6], [l.name for l in chain.links]


def full_link_points(chain: Chain, movable: list[int], q_seq: np.ndarray) -> np.ndarray:
    t_count = len(q_seq)
    l_count = len(chain.links)
    out = np.zeros((t_count, l_count, 3), dtype=np.float64)
    qfull = np.zeros((l_count,), dtype=np.float64)
    for t in range(t_count):
        qfull[:] = 0.0
        for j, idx in enumerate(movable):
            if j < q_seq.shape[1]:
                qfull[idx] = q_seq[t, j]
        mats = chain.forward_kinematics(qfull, full_kinematics=True)
        out[t] = np.asarray(mats)[:, :3, 3]
    return out


def seg_dist(a0, a1, b0, b1) -> float:
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


def collision_stats(link_pts: np.ndarray, link_radius: float, link_names: list[str]):
    t_count, l_count, _ = link_pts.shape
    seg_count = l_count - 1
    pairs = [(i, j) for i in range(seg_count) for j in range(i + 1, seg_count) if abs(i - j) > 1]

    rows = []
    pair_counter = Counter()
    min_d = math.inf
    min_pair = None
    min_margin = math.inf

    for t in range(t_count):
        pts = link_pts[t]
        collided = []
        closest_d = math.inf
        closest_pair = None
        for i, j in pairs:
            d = seg_dist(pts[i], pts[i + 1], pts[j], pts[j + 1])
            if d < closest_d:
                closest_d = d
                closest_pair = (i, j)
            if d < min_d:
                min_d = d
                min_pair = (i, j)
            margin = d - 2.0 * link_radius
            if margin < min_margin:
                min_margin = margin
            if d < 2.0 * link_radius:
                collided.append((i, j))
                pair_counter[(i, j)] += 1
        rows.append(
            {
                "frame": t,
                "collision": int(len(collided) > 0),
                "num_pairs": len(collided),
                "pairs": collided,
                "closest_dist_m": float(closest_d),
                "closest_pair": closest_pair,
            }
        )

    coll_frames = sum(r["collision"] for r in rows)
    summary = {
        "frames_total": t_count,
        "frames_colliding": int(coll_frames),
        "collision_rate": float(coll_frames / max(t_count, 1)),
        "link_radius_m": float(link_radius),
        "overall_min_segment_distance_m": float(min_d),
        "overall_min_margin_m": float(min_margin),
        "overall_min_pair_segments": min_pair,
        "overall_min_pair_links": None
        if min_pair is None
        else [link_names[min_pair[0]], link_names[min_pair[1]]],
        "top_collision_pairs": [
            {
                "pair_segments": [i, j],
                "pair_links": [link_names[i], link_names[j]],
                "frames": int(c),
            }
            for (i, j), c in pair_counter.most_common(10)
        ],
    }
    return rows, summary


def set_equal_3d(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    half = max(float(np.max(maxs - mins) / 2.0), 1e-3) * 1.25
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def write_reports(rows: list[dict], summary: dict, json_path: Path, csv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "collision", "num_pairs", "closest_dist_m", "pairs"])
        for r in rows:
            w.writerow(
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

    LeRobotDataset = bootstrap_lerobot(args.lerobot_src)
    ep_idx_all, action_all, action_names, camera_key = load_metadata(args.dataset_root)
    valid_eps = sorted(np.unique(ep_idx_all).tolist())
    if args.episode not in valid_eps:
        raise ValueError(f"Episode {args.episode} not found. Available: {valid_eps}")

    global_idx = np.where(ep_idx_all == args.episode)[0][:: args.stride]
    action = action_all[global_idx]

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        download_videos=False,
        video_backend="pyav",
    )

    # Decode aligned frames
    frames = []
    for gi in global_idx:
        item = ds[int(gi)]
        img = item[camera_key].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0.0, 1.0)
        frames.append(img)
    frames = np.asarray(frames, dtype=np.float32)

    chain, movable, link_names = build_chain(args.urdf)
    link_pts = full_link_points(chain, movable, action[:, :6])
    
    rows, summary = collision_stats(link_pts, args.link_radius, link_names)

    print(
        f"episode={args.episode} frames={len(frames)} stride={args.stride} "
        f"collision_rate={summary['collision_rate']:.3f}"
    )

    # Render side-by-side GIF
    t_count, l_count, _ = link_pts.shape
    seg_count = l_count - 1
    tcp = link_pts[:, -1, :]

    fig = plt.figure(figsize=(14, 6), dpi=120)
    ax_img = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    img_artist = ax_img.imshow(frames[0])
    ax_img.set_title("True video frame")
    ax_img.axis("off")

    seg_lines = []
    for _ in range(seg_count):
        (ln,) = ax3d.plot([], [], [], lw=3.0, color="tab:blue")
        seg_lines.append(ln)
    (joints_sc,) = ax3d.plot([], [], [], "o", color="k", markersize=4)
    (trail_line,) = ax3d.plot([], [], [], lw=1.5, color="tab:gray", alpha=0.8)
    text3d = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes, va="top")

    set_equal_3d(ax3d, link_pts.reshape(-1, 3))
    ax3d.set_title("URDF kinematic replay + collision (approx)")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")

    def update(i: int):
        img_artist.set_data(frames[i])
        pts = link_pts[i]
        row = rows[i]

        coll_seg = set()
        for a, b in row["pairs"]:
            coll_seg.add(a)
            coll_seg.add(b)

        for s in range(seg_count):
            p0 = pts[s]
            p1 = pts[s + 1]
            seg_lines[s].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            seg_lines[s].set_3d_properties([p0[2], p1[2]])
            seg_lines[s].set_color("crimson" if s in coll_seg else "tab:blue")

        joints_sc.set_data(pts[:, 0], pts[:, 1])
        joints_sc.set_3d_properties(pts[:, 2])

        t0 = max(0, i - args.trail + 1)
        tr = tcp[t0 : i + 1]
        trail_line.set_data(tr[:, 0], tr[:, 1])
        trail_line.set_3d_properties(tr[:, 2])

        collision_text = (
            f"COLLISION pairs={row['num_pairs']}" if row["collision"] else "No collision (approx)"
        )
        text3d.set_text(
            f"frame={i+1}/{t_count}\n{collision_text}\n"
            f"closest_dist={row['closest_dist_m']:.4f} m\n"
            f"gripper={action[i,5]:.3f}"
        )

        return [img_artist, *seg_lines, joints_sc, trail_line, text3d]

    anim = FuncAnimation(fig, update, frames=t_count, interval=1000 / args.fps, blit=False)
    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output_gif), writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    write_reports(rows, summary, args.output_json, args.output_csv)
    print(f"Wrote GIF: {args.output_gif}")
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
