#!/usr/bin/env python3
"""Replay retargeted SO-101 trajectories on a real LeRobot SO-101 arm.

This script is designed for datasets generated in this workspace, where actions
are stored as:
    [joint_1, joint_2, joint_3, joint_4, joint_5, gripper]

It remaps to Feetech SO-101 motor commands:
    shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
    wrist_flex.pos, wrist_roll.pos, gripper.pos

Safety defaults:
- Dry-run by default (no motor command sent). Use --execute to move hardware.
- Optional per-step relative target clamp via max_relative_target.
- Optional smooth move to first action before replay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ARM_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
ALL_MOTORS = ARM_MOTORS + ["gripper"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "lerobot_umi_pick_cube_so101_v3"
DEFAULT_LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
DEFAULT_ORIENTATION_META_NAME = "retarget_orientation.json"
DEFAULT_REST_ARM_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_REST_GRIPPER = 100.0
DEFAULT_ARM_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0]
DEFAULT_ARM_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_LIVE_URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_new_calib.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a retargeted LeRobot dataset episode on a real SO-101 arm."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local LeRobot v3 dataset root (contains data/ and meta/).",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Replay every N-th frame (N>=1). Use >1 to slow/trim trajectory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Replay rate. Default: dataset fps from meta/info.json.",
    )
    parser.add_argument(
        "--arm-unit",
        choices=["auto", "radians", "degrees"],
        default="auto",
        help="Unit of first 5 action channels in dataset.",
    )
    parser.add_argument(
        "--gripper-unit",
        choices=["auto", "zero_one", "percent"],
        default="auto",
        help="Unit of gripper action channel.",
    )
    parser.add_argument(
        "--invert-gripper",
        action="store_true",
        help="Invert gripper command direction.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command motors. Without this flag the script is dry-run only.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem5A460814411",
        help="Serial port for the SO-101 Feetech controller.",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="so101_real",
        help="Robot id used by LeRobot for calibration file naming.",
    )
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Call robot.connect(calibrate=False). Use only if already calibrated.",
    )
    parser.add_argument(
        "--go-to-start-seconds",
        type=float,
        default=2.0,
        help="Smoothly interpolate from current pose to first replay action over this duration.",
    )
    parser.add_argument(
        "--hold-start-seconds",
        type=float,
        default=0.5,
        help="Time to hold at first replay action before starting replay.",
    )
    parser.add_argument(
        "--go-to-rest-seconds",
        type=float,
        default=2.5,
        help="Duration to move from current pose to SO-101 rest pose before replay.",
    )
    parser.add_argument(
        "--skip-rest",
        action="store_true",
        help="Skip moving to rest pose before replay (not recommended).",
    )
    parser.add_argument(
        "--replay-from-start",
        action="store_true",
        help="Replay again from frame 0 after moving to first action. Default is to start from frame 1.",
    )
    parser.add_argument(
        "--rest-arm-deg",
        type=str,
        default="0,0,0,0,0",
        help="SO-101 rest arm joint targets in degrees: shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll.",
    )
    parser.add_argument(
        "--rest-gripper",
        type=float,
        default=DEFAULT_REST_GRIPPER,
        help="SO-101 rest gripper target in percent [0,100].",
    )
    parser.add_argument(
        "--arm-signs",
        type=str,
        default="1,1,1,1,1",
        help="Data-space per-joint sign correction: five comma-separated values.",
    )
    parser.add_argument(
        "--arm-offsets-deg",
        type=str,
        default="0,0,0,0,0",
        help="Data-space per-joint additive offsets in degrees: five comma-separated values.",
    )
    parser.add_argument(
        "--hardware-arm-signs",
        type=str,
        default="1,1,1,1,1",
        help="Hardware-space per-joint sign correction: five comma-separated values.",
    )
    parser.add_argument(
        "--hardware-arm-offsets-deg",
        type=str,
        default="0,0,0,0,0",
        help="Hardware-space per-joint additive offsets in degrees: five comma-separated values.",
    )
    parser.add_argument(
        "--flip-z",
        action="store_true",
        help=(
            "Apply a hardware z-direction flip preset by negating joints "
            "2,3,4 (shoulder_lift, elbow_flex, wrist_flex)."
        ),
    )
    parser.add_argument(
        "--orientation-source",
        choices=["auto", "metadata", "cli"],
        default="auto",
        help=(
            "How to choose arm orientation correction. "
            "'auto' uses metadata if available unless CLI values are explicitly set."
        ),
    )
    parser.add_argument(
        "--orientation-meta",
        type=Path,
        default=None,
        help="Optional explicit path to retarget orientation metadata JSON.",
    )
    parser.add_argument(
        "--max-relative-target-deg",
        type=float,
        default=5.0,
        help="Per-step max change for arm joints in degrees. <=0 disables clamping.",
    )
    parser.add_argument(
        "--max-relative-target-gripper",
        type=float,
        default=12.0,
        help="Per-step max change for gripper in percent [0,100]. <=0 disables clamping.",
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=DEFAULT_LEROBOT_SRC,
        help="Path to lerobot/src. Used when lerobot is not pip-installed.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print progress every N replayed frames.",
    )
    parser.add_argument(
        "--live-motion",
        action="store_true",
        help="Show live 3D URDF TCP motion (commanded vs observed) during replay.",
    )
    parser.add_argument(
        "--live-urdf",
        type=Path,
        default=DEFAULT_LIVE_URDF,
        help="URDF used by --live-motion.",
    )
    parser.add_argument(
        "--live-motion-every",
        type=int,
        default=1,
        help="Update live motion viewer every N replayed frames.",
    )
    parser.add_argument(
        "--live-motion-trail",
        type=int,
        default=180,
        help="History length for live TCP trails.",
    )
    return parser.parse_args()


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")
    return json.loads(info_path.read_text())


def load_episode_actions(dataset_root: Path, episode: int) -> tuple[np.ndarray, list[str], float]:
    info = load_info(dataset_root)
    action_feature = info.get("features", {}).get("action")
    if action_feature is None:
        raise KeyError("Dataset meta/info.json is missing action feature.")
    action_names = action_feature.get("names")
    if not action_names:
        action_names = [f"joint_{i+1}" for i in range(6)]

    fps = float(info.get("fps", 30))

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing dataset data directory: {data_dir}")

    table = pq.ParquetDataset(str(data_dir)).read(columns=["episode_index", "action"])
    dct = table.to_pydict()
    ep_idx = np.asarray(dct["episode_index"], dtype=np.int64)
    action = np.asarray(dct["action"], dtype=np.float64)
    if action.ndim != 2:
        raise ValueError(f"Expected action to be rank-2 [N,D], got shape={action.shape}")
    if action.shape[1] < 6:
        raise ValueError(f"Expected at least 6 action channels, got shape={action.shape}")

    mask = ep_idx == episode
    if not np.any(mask):
        available = sorted(np.unique(ep_idx).tolist())
        raise ValueError(f"Episode {episode} not found. Available episodes: {available}")

    return action[mask, :], list(action_names), fps


def canonical_name(name: str) -> str:
    n = name.strip().lower()
    if n.endswith(".pos"):
        n = n[:-4]
    return n


def infer_source_indices(action_names: list[str]) -> dict[str, int]:
    alias = {
        "joint_1": "shoulder_pan",
        "joint_2": "shoulder_lift",
        "joint_3": "elbow_flex",
        "joint_4": "wrist_flex",
        "joint_5": "wrist_roll",
        "gripper": "gripper",
        "shoulder_pan": "shoulder_pan",
        "shoulder_lift": "shoulder_lift",
        "elbow_flex": "elbow_flex",
        "wrist_flex": "wrist_flex",
        "wrist_roll": "wrist_roll",
    }

    out: dict[str, int] = {}
    for i, raw in enumerate(action_names):
        key = canonical_name(raw)
        if key in alias:
            out[alias[key]] = i

    missing = [m for m in ALL_MOTORS if m not in out]
    if missing:
        raise ValueError(
            f"Could not map all required action channels. Missing={missing}, action_names={action_names}"
        )
    return out


def detect_arm_unit(actions: np.ndarray, arm_unit: str) -> str:
    if arm_unit != "auto":
        return arm_unit
    max_abs = float(np.max(np.abs(actions[:, :5])))
    if max_abs <= 3.5:
        return "radians"
    return "degrees"


def detect_gripper_unit(actions: np.ndarray, gripper_idx: int, gripper_unit: str) -> str:
    if gripper_unit != "auto":
        return gripper_unit
    g = actions[:, gripper_idx]
    gmin = float(np.min(g))
    gmax = float(np.max(g))
    if gmin >= -0.05 and gmax <= 1.05:
        return "zero_one"
    if gmin >= -1.0 and gmax <= 101.0:
        return "percent"
    raise ValueError(
        f"Unable to infer gripper unit from range [{gmin:.4f}, {gmax:.4f}]. "
        "Specify --gripper-unit explicitly."
    )


def build_motor_action(
    action_row: np.ndarray,
    src_indices: dict[str, int],
    arm_unit: str,
    gripper_unit: str,
    invert_gripper: bool,
    arm_signs: np.ndarray,
    arm_offsets_deg: np.ndarray,
) -> dict[str, float]:
    arm_vals = np.zeros((5,), dtype=np.float64)
    for j, motor in enumerate(ARM_MOTORS):
        val = float(action_row[src_indices[motor]])
        if arm_unit == "radians":
            val = float(np.degrees(val))
        arm_vals[j] = val
    arm_vals = arm_vals * arm_signs + arm_offsets_deg

    out: dict[str, float] = {}
    for j, motor in enumerate(ARM_MOTORS):
        out[f"{motor}.pos"] = float(arm_vals[j])

    g = float(action_row[src_indices["gripper"]])
    if gripper_unit == "zero_one":
        g *= 100.0
    g = float(np.clip(g, 0.0, 100.0))
    if invert_gripper:
        g = 100.0 - g
    out["gripper.pos"] = g
    return out


def bootstrap_lerobot(lerobot_src: Path) -> None:
    if "lerobot" in sys.modules:
        return
    if lerobot_src.exists() and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def build_max_relative_target(args: argparse.Namespace) -> float | dict[str, float] | None:
    arm = args.max_relative_target_deg
    grip = args.max_relative_target_gripper
    if arm <= 0 and grip <= 0:
        return None
    if arm > 0 and grip > 0:
        return {
            "shoulder_pan": arm,
            "shoulder_lift": arm,
            "elbow_flex": arm,
            "wrist_flex": arm,
            "wrist_roll": arm,
            "gripper": grip,
        }
    if arm > 0:
        return float(arm)
    return {"gripper": float(grip)}


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def parse_five_floats(raw: str, label: str, default: list[float]) -> np.ndarray:
    txt = raw.strip()
    if txt == "":
        return np.asarray(default, dtype=np.float64)
    parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
    if len(parts) != 5:
        raise ValueError(f"{label} must contain exactly 5 comma-separated values, got: {raw!r}")
    try:
        return np.asarray([float(x) for x in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Invalid float values for {label}: {raw!r}") from exc


def parse_five_from_meta(meta: dict[str, Any], key: str) -> np.ndarray:
    if key not in meta:
        raise KeyError(f"Orientation metadata missing key: {key}")
    vals = np.asarray(meta[key], dtype=np.float64).reshape(-1)
    if vals.shape != (5,):
        raise ValueError(f"Orientation metadata key '{key}' must have 5 values, got {vals.shape}")
    return vals


def load_orientation_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text())
    # Validate expected keys when present.
    if "replay_joint_signs" in data:
        _ = parse_five_from_meta(data, "replay_joint_signs")
    if "replay_joint_offsets_deg" in data:
        _ = parse_five_from_meta(data, "replay_joint_offsets_deg")
    if "dataset_joint_signs" in data:
        _ = parse_five_from_meta(data, "dataset_joint_signs")
    if "dataset_joint_offsets_deg" in data:
        _ = parse_five_from_meta(data, "dataset_joint_offsets_deg")
    return data


def correction_from_orientation_meta(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    has_replay = "replay_joint_signs" in meta or "replay_joint_offsets_deg" in meta
    if has_replay:
        signs = (
            parse_five_from_meta(meta, "replay_joint_signs")
            if "replay_joint_signs" in meta
            else np.asarray(DEFAULT_ARM_SIGNS, dtype=np.float64)
        )
        offsets = (
            parse_five_from_meta(meta, "replay_joint_offsets_deg")
            if "replay_joint_offsets_deg" in meta
            else np.asarray(DEFAULT_ARM_OFFSETS_DEG, dtype=np.float64)
        )
        return signs, offsets, "replay_joint_*"

    # Backward-compatible behavior for older metadata:
    # if data was already orientation-corrected before writing, replay should be identity.
    if bool(meta.get("applied_in_dataset", False)):
        return (
            np.ones((5,), dtype=np.float64),
            np.zeros((5,), dtype=np.float64),
            "applied_in_dataset(identity)",
        )

    if "dataset_joint_signs" in meta or "dataset_joint_offsets_deg" in meta:
        signs = (
            parse_five_from_meta(meta, "dataset_joint_signs")
            if "dataset_joint_signs" in meta
            else np.asarray(DEFAULT_ARM_SIGNS, dtype=np.float64)
        )
        offsets = (
            parse_five_from_meta(meta, "dataset_joint_offsets_deg")
            if "dataset_joint_offsets_deg" in meta
            else np.asarray(DEFAULT_ARM_OFFSETS_DEG, dtype=np.float64)
        )
        return signs, offsets, "dataset_joint_*"

    return (
        np.asarray(DEFAULT_ARM_SIGNS, dtype=np.float64),
        np.asarray(DEFAULT_ARM_OFFSETS_DEG, dtype=np.float64),
        "defaults",
    )


def resolve_orientation_correction(
    args: argparse.Namespace,
    dataset_root: Path,
    cli_signs: np.ndarray,
    cli_offsets_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    meta_path = args.orientation_meta
    if meta_path is None:
        meta_path = dataset_root / "meta" / DEFAULT_ORIENTATION_META_NAME
    meta = load_orientation_meta(meta_path)

    cli_default_signs = np.asarray(DEFAULT_ARM_SIGNS, dtype=np.float64)
    cli_default_offsets = np.asarray(DEFAULT_ARM_OFFSETS_DEG, dtype=np.float64)
    cli_is_explicit = not (
        np.allclose(cli_signs, cli_default_signs) and np.allclose(cli_offsets_deg, cli_default_offsets)
    )

    if args.orientation_source == "cli":
        return cli_signs, cli_offsets_deg, "cli"

    if args.orientation_source == "metadata":
        if meta is None:
            raise FileNotFoundError(
                f"--orientation-source=metadata but metadata file not found: {meta_path}"
            )
        signs, offsets, key_source = correction_from_orientation_meta(meta)
        return signs, offsets, f"metadata({key_source}):{meta_path}"

    # auto
    if cli_is_explicit:
        return cli_signs, cli_offsets_deg, "cli(explicit)"
    if meta is not None:
        signs, offsets, key_source = correction_from_orientation_meta(meta)
        return signs, offsets, f"metadata(auto,{key_source}):{meta_path}"
    return cli_signs, cli_offsets_deg, "cli(default)"


@dataclass
class LiveMotionView:
    fig: Any
    ax3d: Any
    axz: Any
    cmd_line_3d: Any
    obs_line_3d: Any
    cmd_point_3d: Any
    obs_point_3d: Any
    cmd_line_z: Any
    obs_line_z: Any
    txt_status: Any
    chain: Any
    movable_indices: list[int]
    trail: int
    frames: list[int]
    cmd_traj: list[np.ndarray]
    obs_traj: list[np.ndarray]
    z_cmd: list[float]
    z_obs: list[float]


def create_live_motion_view(urdf_path: Path, trail: int) -> LiveMotionView:
    if not urdf_path.exists():
        raise FileNotFoundError(f"--live-motion requested but URDF not found: {urdf_path}")
    try:
        import matplotlib.pyplot as plt
        from ikpy.chain import Chain
    except Exception as exc:
        raise ModuleNotFoundError(
            "--live-motion requires matplotlib and ikpy in this environment."
        ) from exc

    warnings.filterwarnings("ignore", category=UserWarning, module="ikpy")
    chain = Chain.from_urdf_file(str(urdf_path))
    movable = [i for i, link in enumerate(chain.links) if getattr(link, "joint_type", "fixed") != "fixed"]
    if len(movable) < 5:
        raise ValueError(f"Need >=5 movable joints in URDF for live motion, got {len(movable)}")
    movable = movable[:5]

    plt.ion()
    fig = plt.figure(figsize=(11, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axz = fig.add_subplot(1, 2, 2)
    fig.suptitle("SO-101 live replay debug")

    cmd_line_3d, = ax3d.plot([], [], [], "b-", lw=2, label="cmd TCP")
    obs_line_3d, = ax3d.plot([], [], [], "g-", lw=2, label="obs TCP")
    cmd_point_3d, = ax3d.plot([], [], [], "bo", ms=6)
    obs_point_3d, = ax3d.plot([], [], [], "go", ms=6)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D TCP trajectory")
    ax3d.legend(loc="upper left")

    cmd_line_z, = axz.plot([], [], "b-", lw=2, label="cmd Z")
    obs_line_z, = axz.plot([], [], "g-", lw=2, label="obs Z")
    axz.set_xlabel("Replay frame")
    axz.set_ylabel("Z (m)")
    axz.set_title("Z over time")
    axz.grid(True, alpha=0.25)
    axz.legend(loc="best")

    txt_status = fig.text(0.5, 0.01, "", ha="center", va="bottom")
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.show(block=False)

    return LiveMotionView(
        fig=fig,
        ax3d=ax3d,
        axz=axz,
        cmd_line_3d=cmd_line_3d,
        obs_line_3d=obs_line_3d,
        cmd_point_3d=cmd_point_3d,
        obs_point_3d=obs_point_3d,
        cmd_line_z=cmd_line_z,
        obs_line_z=obs_line_z,
        txt_status=txt_status,
        chain=chain,
        movable_indices=movable,
        trail=max(int(trail), 10),
        frames=[],
        cmd_traj=[],
        obs_traj=[],
        z_cmd=[],
        z_obs=[],
    )


def fk_tcp_from_arm_deg(chain: Any, movable_indices: list[int], arm_deg: np.ndarray) -> np.ndarray:
    q_full = np.zeros((len(chain.links),), dtype=np.float64)
    arm_rad = np.deg2rad(np.asarray(arm_deg[:5], dtype=np.float64))
    for j, idx in enumerate(movable_indices):
        if j < arm_rad.shape[0]:
            q_full[idx] = arm_rad[j]
    T = chain.forward_kinematics(q_full)
    return np.asarray(T[:3, 3], dtype=np.float64)


def update_live_motion_view(
    live: LiveMotionView,
    frame_idx: int,
    cmd_vec_deg: np.ndarray,
    obs_vec_deg: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    cmd_tcp = fk_tcp_from_arm_deg(live.chain, live.movable_indices, cmd_vec_deg)
    obs_tcp = fk_tcp_from_arm_deg(live.chain, live.movable_indices, obs_vec_deg)

    live.frames.append(int(frame_idx))
    live.cmd_traj.append(cmd_tcp)
    live.obs_traj.append(obs_tcp)
    live.z_cmd.append(float(cmd_tcp[2]))
    live.z_obs.append(float(obs_tcp[2]))

    if len(live.frames) > live.trail:
        live.frames.pop(0)
        live.cmd_traj.pop(0)
        live.obs_traj.pop(0)
        live.z_cmd.pop(0)
        live.z_obs.pop(0)

    cmd_arr = np.asarray(live.cmd_traj)
    obs_arr = np.asarray(live.obs_traj)
    all_arr = np.vstack([cmd_arr, obs_arr])

    live.cmd_line_3d.set_data(cmd_arr[:, 0], cmd_arr[:, 1])
    live.cmd_line_3d.set_3d_properties(cmd_arr[:, 2])
    live.obs_line_3d.set_data(obs_arr[:, 0], obs_arr[:, 1])
    live.obs_line_3d.set_3d_properties(obs_arr[:, 2])
    live.cmd_point_3d.set_data([cmd_arr[-1, 0]], [cmd_arr[-1, 1]])
    live.cmd_point_3d.set_3d_properties([cmd_arr[-1, 2]])
    live.obs_point_3d.set_data([obs_arr[-1, 0]], [obs_arr[-1, 1]])
    live.obs_point_3d.set_3d_properties([obs_arr[-1, 2]])

    margin = 0.03
    xmin, ymin, zmin = np.min(all_arr, axis=0) - margin
    xmax, ymax, zmax = np.max(all_arr, axis=0) + margin
    if np.isclose(xmin, xmax):
        xmin, xmax = xmin - 0.05, xmax + 0.05
    if np.isclose(ymin, ymax):
        ymin, ymax = ymin - 0.05, ymax + 0.05
    if np.isclose(zmin, zmax):
        zmin, zmax = zmin - 0.05, zmax + 0.05
    live.ax3d.set_xlim(xmin, xmax)
    live.ax3d.set_ylim(ymin, ymax)
    live.ax3d.set_zlim(zmin, zmax)

    f = np.asarray(live.frames, dtype=np.int64)
    live.cmd_line_z.set_data(f, np.asarray(live.z_cmd, dtype=np.float64))
    live.obs_line_z.set_data(f, np.asarray(live.z_obs, dtype=np.float64))
    live.axz.set_xlim(max(int(f[0]), int(f[-1]) - live.trail), int(f[-1]) + 1)
    z_join = np.asarray(live.z_cmd + live.z_obs, dtype=np.float64)
    z_lo = float(np.min(z_join) - 0.02)
    z_hi = float(np.max(z_join) + 0.02)
    if np.isclose(z_lo, z_hi):
        z_lo, z_hi = z_lo - 0.05, z_hi + 0.05
    live.axz.set_ylim(z_lo, z_hi)

    dz = float(obs_tcp[2] - cmd_tcp[2])
    live.txt_status.set_text(f"frame={frame_idx} | cmd_z={cmd_tcp[2]:+.3f}m obs_z={obs_tcp[2]:+.3f}m dz={dz:+.3f}m")
    live.fig.canvas.draw_idle()
    live.fig.canvas.flush_events()
    plt.pause(0.001)


def close_live_motion_view(live: LiveMotionView | None) -> None:
    if live is None:
        return
    try:
        import matplotlib.pyplot as plt

        plt.close(live.fig)
    except Exception:
        return


def move_linear(
    robot,
    mapped_names: list[str],
    start: np.ndarray,
    target: np.ndarray,
    seconds: float,
    fps: float,
    label: str,
) -> None:
    if seconds <= 0:
        cmd = {k: float(v) for k, v in zip(mapped_names, target, strict=True)}
        robot.send_action(cmd)
        precise_sleep(0.2)
        return

    n_steps = max(int(seconds * fps), 2)
    print(f"{label} over {seconds:.2f}s ({n_steps} interpolation steps)...")
    for i in range(1, n_steps + 1):
        a = i / n_steps
        cmd_vec = (1.0 - a) * start + a * target
        cmd = {k: float(v) for k, v in zip(mapped_names, cmd_vec, strict=True)}
        robot.send_action(cmd)
        precise_sleep(1.0 / fps)


def print_preview(mapped_actions: np.ndarray, mapped_names: list[str], fps: float, is_execute: bool) -> None:
    print("Replay summary")
    print(f"  execute: {is_execute}")
    print(f"  frames: {mapped_actions.shape[0]}")
    print(f"  command fps: {fps:.3f}")
    print(f"  action order: {mapped_names}")
    print("  first action:", np.round(mapped_actions[0], 4).tolist())
    print("  last action: ", np.round(mapped_actions[-1], 4).tolist())
    mins = np.min(mapped_actions, axis=0)
    maxs = np.max(mapped_actions, axis=0)
    print("  min:", np.round(mins, 4).tolist())
    print("  max:", np.round(maxs, 4).tolist())


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.print_every < 1:
        raise ValueError("--print-every must be >= 1")
    if args.live_motion_every < 1:
        raise ValueError("--live-motion-every must be >= 1")
    if args.hold_start_seconds < 0:
        raise ValueError("--hold-start-seconds must be >= 0")

    actions, action_names, dataset_fps = load_episode_actions(args.dataset_root, args.episode)
    src_indices = infer_source_indices(action_names)
    arm_unit = detect_arm_unit(actions, args.arm_unit)
    gripper_unit = detect_gripper_unit(actions, src_indices["gripper"], args.gripper_unit)
    cli_arm_signs = parse_five_floats(args.arm_signs, "--arm-signs", DEFAULT_ARM_SIGNS)
    cli_arm_offsets_deg = parse_five_floats(args.arm_offsets_deg, "--arm-offsets-deg", DEFAULT_ARM_OFFSETS_DEG)
    data_arm_signs, data_arm_offsets_deg, orientation_source = resolve_orientation_correction(
        args=args,
        dataset_root=args.dataset_root,
        cli_signs=cli_arm_signs,
        cli_offsets_deg=cli_arm_offsets_deg,
    )
    hardware_arm_signs = parse_five_floats(
        args.hardware_arm_signs, "--hardware-arm-signs", DEFAULT_ARM_SIGNS
    )
    hardware_arm_offsets_deg = parse_five_floats(
        args.hardware_arm_offsets_deg, "--hardware-arm-offsets-deg", DEFAULT_ARM_OFFSETS_DEG
    )
    if args.flip_z:
        # Hardware-specific quick fix for "vertical motion appears inverted":
        # invert shoulder_lift, elbow_flex, wrist_flex.
        hardware_arm_signs[1:4] *= -1.0
    rest_arm_deg = parse_five_floats(args.rest_arm_deg, "--rest-arm-deg", DEFAULT_REST_ARM_DEG)
    rest_gripper = float(np.clip(args.rest_gripper, 0.0, 100.0))

    replay_fps = float(args.fps if args.fps is not None else dataset_fps)
    if replay_fps <= 0:
        raise ValueError("Replay fps must be > 0")

    # Pre-map all frames once for deterministic preview and runtime speed.
    mapped = []
    for row in actions[:: args.stride]:
        cmd = build_motor_action(
            action_row=row,
            src_indices=src_indices,
            arm_unit=arm_unit,
            gripper_unit=gripper_unit,
            invert_gripper=args.invert_gripper,
            arm_signs=data_arm_signs,
            arm_offsets_deg=data_arm_offsets_deg,
        )
        cmd["shoulder_pan.pos"] = cmd["shoulder_pan.pos"] * hardware_arm_signs[0] + hardware_arm_offsets_deg[0]
        cmd["shoulder_lift.pos"] = cmd["shoulder_lift.pos"] * hardware_arm_signs[1] + hardware_arm_offsets_deg[1]
        cmd["elbow_flex.pos"] = cmd["elbow_flex.pos"] * hardware_arm_signs[2] + hardware_arm_offsets_deg[2]
        cmd["wrist_flex.pos"] = cmd["wrist_flex.pos"] * hardware_arm_signs[3] + hardware_arm_offsets_deg[3]
        cmd["wrist_roll.pos"] = cmd["wrist_roll.pos"] * hardware_arm_signs[4] + hardware_arm_offsets_deg[4]
        mapped.append(
            [
                cmd["shoulder_pan.pos"],
                cmd["shoulder_lift.pos"],
                cmd["elbow_flex.pos"],
                cmd["wrist_flex.pos"],
                cmd["wrist_roll.pos"],
                cmd["gripper.pos"],
            ]
        )
    mapped_actions = np.asarray(mapped, dtype=np.float64)
    mapped_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    print(f"Inferred units: arm={arm_unit}, gripper={gripper_unit}")
    print(
        "Data orientation correction: "
        f"signs={np.round(data_arm_signs, 3).tolist()} "
        f"offsets_deg={np.round(data_arm_offsets_deg, 3).tolist()}"
    )
    print(
        "Hardware orientation correction: "
        f"signs={np.round(hardware_arm_signs, 3).tolist()} "
        f"offsets_deg={np.round(hardware_arm_offsets_deg, 3).tolist()} "
        f"(flip_z={args.flip_z})"
    )
    print(f"Orientation source: {orientation_source}")
    print(
        "Rest pose: "
        f"arm_deg={np.round(rest_arm_deg, 3).tolist()} "
        f"gripper={rest_gripper:.1f}"
    )
    print_preview(mapped_actions, mapped_names, replay_fps, args.execute)

    if not args.execute:
        print("\nDry-run complete. Use --execute to send commands to motors.")
        return 0

    bootstrap_lerobot(args.lerobot_src)
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except Exception as exc:  # pragma: no cover - hardware env dependent
        raise ModuleNotFoundError(
            "Failed to import lerobot SO-101 classes.\n"
            f"Tried lerobot src path: {args.lerobot_src}\n"
            "Install/activate environment with lerobot + feetech-servo-sdk, "
            "or pass --lerobot-src to a valid clone."
        ) from exc

    max_relative_target = build_max_relative_target(args)
    config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=max_relative_target,
    )
    robot = SO101Follower(config)

    print("\nConnecting to robot...")
    robot.connect(calibrate=not args.skip_calibrate)
    if not robot.is_connected:
        raise RuntimeError("Robot failed to connect.")

    live: LiveMotionView | None = None
    try:
        if args.live_motion:
            live = create_live_motion_view(args.live_urdf, args.live_motion_trail)
            print(f"Live motion viewer enabled with URDF: {args.live_urdf}")

        obs = robot.get_observation()
        now = np.array([float(obs[k]) for k in mapped_names], dtype=np.float64)

        if not args.skip_rest:
            rest_target = np.concatenate([rest_arm_deg, np.array([rest_gripper], dtype=np.float64)])
            move_linear(
                robot=robot,
                mapped_names=mapped_names,
                start=now,
                target=rest_target,
                seconds=args.go_to_rest_seconds,
                fps=replay_fps,
                label="Moving to SO-101 rest pose",
            )
            now = rest_target
        else:
            print("Skipping rest pose move (--skip-rest was set).")

        first_target = mapped_actions[0]
        move_linear(
            robot=robot,
            mapped_names=mapped_names,
            start=now,
            target=first_target,
            seconds=args.go_to_start_seconds,
            fps=replay_fps,
            label="Moving to first replay action",
        )

        if args.hold_start_seconds > 0:
            print(f"Holding at start pose for {args.hold_start_seconds:.2f}s...")
            precise_sleep(args.hold_start_seconds)

        print("\nStarting replay...")
        total = mapped_actions.shape[0]
        replay_start_idx = 0 if args.replay_from_start else 1
        if replay_start_idx >= total:
            replay_start_idx = 0
        if replay_start_idx == 1:
            print("Replay starts from frame 2 (frame 1 used as start pose).")
        for i, cmd_vec in enumerate(mapped_actions[replay_start_idx:], start=replay_start_idx + 1):
            t0 = time.perf_counter()
            cmd = {k: float(v) for k, v in zip(mapped_names, cmd_vec, strict=True)}
            robot.send_action(cmd)

            if live is not None and ((i - replay_start_idx) % args.live_motion_every == 0 or i == total):
                obs_now = robot.get_observation()
                obs_vec = np.array([float(obs_now[k]) for k in mapped_names], dtype=np.float64)
                update_live_motion_view(
                    live=live,
                    frame_idx=i,
                    cmd_vec_deg=cmd_vec,
                    obs_vec_deg=obs_vec,
                )

            if i % args.print_every == 0 or i == total:
                print(
                    f"frame {i:4d}/{total} | "
                    f"arm(deg)={np.round(cmd_vec[:5], 2).tolist()} "
                    f"gripper={cmd_vec[5]:.1f}"
                )
            elapsed = time.perf_counter() - t0
            precise_sleep(max(1.0 / replay_fps - elapsed, 0.0))

        print("Replay completed.")
    finally:
        close_live_motion_view(live)
        print("Disconnecting robot...")
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
