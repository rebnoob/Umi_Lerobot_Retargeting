#!/usr/bin/env python3
"""Render a 3D GIF of retargeted 6-DoF action from a LeRobot v3 dataset.

This script expects action = [joint_1..joint_5, gripper].
It uses IKPy forward kinematics on the provided URDF to animate TCP motion.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from ikpy.chain import Chain
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render retargeted 6-DoF motion as GIF.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/datasets/lerobot_umi_pick_cube_so101_v3"),
        help="LeRobot v3 dataset root path.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/assets/urdf/so101_new_calib.urdf"),
        help="SO-101 URDF path used for forward kinematics.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to render.")
    parser.add_argument("--stride", type=int, default=2, help="Use every N-th frame.")
    parser.add_argument("--trail", type=int, default=60, help="Trajectory trail length in frames.")
    parser.add_argument("--fps", type=int, default=20, help="Output GIF fps.")
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.04,
        help="TCP orientation triad axis length (meters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/rebnoob/Downloads/Umi Data/outputs/videos/retargeted_6dof_ep0.gif"),
        help="Output GIF path.",
    )
    return parser.parse_args()


def load_actions(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = json.loads(info_path.read_text())

    action_feature = info.get("features", {}).get("action")
    if action_feature is None:
        raise KeyError("Dataset info.json does not contain 'action' feature.")
    action_names = action_feature.get("names", [f"action_{i}" for i in range(6)])

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    table = pq.ParquetDataset(str(data_dir)).read(columns=["action", "episode_index"])
    dct = table.to_pydict()
    actions = np.asarray(dct["action"], dtype=np.float64)
    episode_index = np.asarray(dct["episode_index"], dtype=np.int64)

    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected action shape [N, >=6], got {actions.shape}")
    return actions[:, :6], episode_index, action_names[:6]


def create_chain(urdf_path: Path) -> tuple[Chain, list[int]]:
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    warnings.filterwarnings("ignore", category=UserWarning, module="ikpy")
    chain = Chain.from_urdf_file(str(urdf_path))

    movable_link_indices = [
        i for i, link in enumerate(chain.links) if getattr(link, "joint_type", "fixed") != "fixed"
    ]
    if len(movable_link_indices) < 5:
        raise ValueError(
            f"URDF has only {len(movable_link_indices)} movable joints, but 5 are required."
        )
    return chain, movable_link_indices[:5]


def fk_trajectory(chain: Chain, movable_indices: list[int], q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = q.shape[0]
    positions = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    q_full = np.zeros((len(chain.links),), dtype=np.float64)

    for i in range(n):
        q_full[:] = 0.0
        for j, link_idx in enumerate(movable_indices):
            q_full[link_idx] = q[i, j]
        t = chain.forward_kinematics(q_full)
        positions[i] = t[:3, 3]
        rotations[i] = t[:3, :3]
    return positions, rotations


def make_equal_3d_limits(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    half = np.max(maxs - mins) / 2.0
    half = max(half, 1e-3)
    pad = 0.15 * half
    half += pad
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render_gif(
    pos: np.ndarray,
    rot: np.ndarray,
    action: np.ndarray,
    action_names: list[str],
    output: Path,
    fps: int,
    trail: int,
    axis_scale: float,
) -> None:
    n = pos.shape[0]
    frames = np.arange(n, dtype=np.int64)

    fig = plt.figure(figsize=(12, 6), dpi=120)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axa = fig.add_subplot(1, 2, 2)

    line, = ax3d.plot([], [], [], lw=2.0, color="tab:blue", label="tcp path")
    point = ax3d.scatter([], [], [], color="red", s=28, label="tcp")
    triad_artists = []

    make_equal_3d_limits(ax3d, pos)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("Retargeted TCP 3D Motion")
    ax3d.legend(loc="upper right")

    x = np.arange(n)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for j in range(6):
        axa.plot(x, action[:, j], lw=1.0, color=colors[j], label=action_names[j])
    vline = axa.axvline(0, color="k", lw=1.2, alpha=0.8)
    frame_txt = axa.text(0.02, 0.95, "", transform=axa.transAxes, va="top")
    axa.set_title("6-DoF Action Channels")
    axa.set_xlabel("frame (episode-local)")
    axa.grid(alpha=0.25)
    axa.legend(loc="upper right", fontsize=8)

    def update(i: int):
        nonlocal triad_artists
        cur = int(frames[i])
        t0 = max(0, cur - trail + 1)
        p = pos[t0 : cur + 1]

        line.set_data(p[:, 0], p[:, 1])
        line.set_3d_properties(p[:, 2])
        point._offsets3d = ([pos[cur, 0]], [pos[cur, 1]], [pos[cur, 2]])

        for artist in triad_artists:
            artist.remove()
        triad_artists = []

        r = rot[cur]
        c = pos[cur]
        for col, clr in zip(range(3), ["r", "g", "b"]):
            v = r[:, col] * axis_scale
            triad_artists.append(
                ax3d.quiver(c[0], c[1], c[2], v[0], v[1], v[2], color=clr, linewidth=2.0)
            )

        vline.set_xdata([cur, cur])
        frame_txt.set_text(f"frame={cur+1}/{n}  gripper={action[cur, 5]:.3f}")
        return [line, point, vline, frame_txt, *triad_artists]

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / max(fps, 1), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.trail < 1:
        raise ValueError("--trail must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")

    actions_all, ep_all, action_names = load_actions(args.dataset_root)
    mask = ep_all == args.episode
    if not np.any(mask):
        raise ValueError(
            f"Episode {args.episode} not found. Available: {sorted(np.unique(ep_all).tolist())}"
        )
    act_ep = actions_all[mask]
    act_ep = act_ep[:: args.stride]

    chain, movable = create_chain(args.urdf)
    q = act_ep[:, :5]
    pos, rot = fk_trajectory(chain, movable, q)

    print(f"Rendering episode={args.episode}, frames={len(act_ep)}, stride={args.stride}")
    print(f"URDF movable joint indices used for q1..q5: {movable}")
    print(f"Output: {args.output}")
    render_gif(
        pos=pos,
        rot=rot,
        action=act_ep,
        action_names=action_names,
        output=args.output,
        fps=args.fps,
        trail=args.trail,
        axis_scale=args.axis_scale,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
