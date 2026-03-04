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
DEFAULT_REST_ARM_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_REST_GRIPPER = 100.0
DEFAULT_ARM_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0]
DEFAULT_ARM_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]


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
        help="Per-joint sign correction for case-specific orientation: five comma-separated values.",
    )
    parser.add_argument(
        "--arm-offsets-deg",
        type=str,
        default="0,0,0,0,0",
        help="Per-joint additive offsets in degrees for case-specific orientation: five comma-separated values.",
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

    actions, action_names, dataset_fps = load_episode_actions(args.dataset_root, args.episode)
    src_indices = infer_source_indices(action_names)
    arm_unit = detect_arm_unit(actions, args.arm_unit)
    gripper_unit = detect_gripper_unit(actions, src_indices["gripper"], args.gripper_unit)
    arm_signs = parse_five_floats(args.arm_signs, "--arm-signs", DEFAULT_ARM_SIGNS)
    arm_offsets_deg = parse_five_floats(args.arm_offsets_deg, "--arm-offsets-deg", DEFAULT_ARM_OFFSETS_DEG)
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
            arm_signs=arm_signs,
            arm_offsets_deg=arm_offsets_deg,
        )
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
        "Case orientation correction: "
        f"signs={np.round(arm_signs, 3).tolist()} "
        f"offsets_deg={np.round(arm_offsets_deg, 3).tolist()}"
    )
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

    try:
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

        print("\nStarting replay...")
        total = mapped_actions.shape[0]
        for i, cmd_vec in enumerate(mapped_actions, start=1):
            t0 = time.perf_counter()
            cmd = {k: float(v) for k, v in zip(mapped_names, cmd_vec, strict=True)}
            robot.send_action(cmd)
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
        print("Disconnecting robot...")
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
