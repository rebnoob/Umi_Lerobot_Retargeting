#!/usr/bin/env python3
"""Interactive and CLI reader for UMI-format Zarr datasets."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numcodecs

try:
    import zarr
except ImportError as exc:  # pragma: no cover - import guard for clearer UX
    raise SystemExit(
        "Missing dependency: zarr.\n"
        "Install dependencies first:\n"
        "  python3 -m pip install zarr numcodecs imagecodecs imagecodecs-numcodecs"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZARR_PATH = PROJECT_ROOT / "data" / "raw" / "pick_cube.zarr"


@dataclass
class Selection:
    mode: str
    index: Optional[int] = None
    start: Optional[int] = None
    stop: Optional[int] = None
    episode: Optional[int] = None
    episode_start: Optional[int] = None
    episode_stop: Optional[int] = None


def register_umi_codecs() -> None:
    """Register codecs used by UMI datasets (notably JPEG-XL)."""
    try:
        numcodecs.get_codec({"id": "imagecodecs_jpegxl"})
        return
    except Exception:
        pass

    errors = []
    for module_name in (
        "diffusion_policy.codecs.imagecodecs_numcodecs",
        "imagecodecs_numcodecs",
    ):
        try:
            module = __import__(module_name, fromlist=["register_codecs"])
            module.register_codecs()
            return
        except Exception as exc:  # pragma: no cover - best effort registration
            errors.append((module_name, str(exc)))

    try:
        numcodecs.get_codec({"id": "imagecodecs_jpegxl"})
        return
    except Exception:
        pass

    error_text = "\n".join(f"- {name}: {msg}" for name, msg in errors)
    print(
        "Warning: could not auto-register image codecs.\n"
        "If you read image arrays (e.g. camera RGB), install one of:\n"
        "  python3 -m pip install imagecodecs imagecodecs-numcodecs\n"
        "or install diffusion_policy.\n"
        f"Details:\n{error_text}",
        file=sys.stderr,
    )


def load_dataset(path: Path):
    warnings.filterwarnings(
        "ignore",
        message=r"Object at \.DS_Store is not recognized as a component of a Zarr hierarchy\.",
    )
    root = zarr.open(str(path), mode="r")
    if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError(
            "This Zarr store does not match expected UMI layout: "
            "needs /data and /meta/episode_ends."
        )
    return root


def compressor_label(arr) -> str:
    compressors = getattr(arr, "compressors", ())
    if not compressors:
        return "none"
    comp = compressors[0]
    if hasattr(comp, "codec_id"):
        return str(comp.codec_id)
    return comp.__class__.__name__


def print_summary(root) -> list[str]:
    data_group = root["data"]
    keys = sorted(data_group.array_keys())
    episode_ends = np.asarray(root["meta"]["episode_ends"])
    print(f"Dataset: {root.store.path if hasattr(root.store, 'path') else '<zarr>'}")
    print(f"Total timesteps: {int(episode_ends[-1]) if len(episode_ends) else 0}")
    print(f"Total episodes: {len(episode_ends)}")
    print("Available /data arrays:")
    for idx, key in enumerate(keys):
        arr = data_group[key]
        print(
            f"  [{idx}] {key:28s} shape={arr.shape} dtype={arr.dtype} "
            f"chunks={arr.chunks} compressor={compressor_label(arr)}"
        )
    return keys


def build_episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    if episode_ends.ndim != 1:
        raise ValueError("/meta/episode_ends must be a 1D array.")
    if len(episode_ends) == 0:
        return []
    if np.any(np.diff(episode_ends) <= 0):
        raise ValueError("/meta/episode_ends must be strictly increasing.")
    starts = np.concatenate(([0], episode_ends[:-1]))
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def prompt_int(message: str, *, low: int, high: int, allow_blank: bool = False) -> Optional[int]:
    while True:
        raw = input(message).strip()
        if allow_blank and raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < low or value > high:
            print(f"Value must be in [{low}, {high}].")
            continue
        return value


def interactive_selection(keys: list[str], episode_bounds: list[tuple[int, int]]) -> tuple[str, Selection]:
    key_idx = prompt_int(
        f"Pick data key index [0-{len(keys) - 1}]: ",
        low=0,
        high=len(keys) - 1,
    )
    key = keys[key_idx]

    print("\nRead mode:")
    print("  [1] Single absolute timestep index")
    print("  [2] Absolute timestep range [start:stop)")
    print("  [3] Episode slice (episode + relative range)")
    mode = prompt_int("Pick mode [1-3]: ", low=1, high=3)

    if mode == 1:
        index = prompt_int("Absolute index: ", low=0, high=episode_bounds[-1][1] - 1)
        return key, Selection(mode="index", index=index)

    if mode == 2:
        total = episode_bounds[-1][1]
        start = prompt_int("Start index: ", low=0, high=total - 1)
        stop = prompt_int("Stop index (exclusive): ", low=start + 1, high=total)
        return key, Selection(mode="range", start=start, stop=stop)

    ep = prompt_int("Episode index: ", low=0, high=len(episode_bounds) - 1)
    ep_start, ep_stop = episode_bounds[ep]
    ep_len = ep_stop - ep_start
    rel_start = prompt_int("Episode relative start (blank = 0): ", low=0, high=ep_len - 1, allow_blank=True)
    rel_stop = prompt_int(
        "Episode relative stop exclusive (blank = end): ",
        low=1,
        high=ep_len,
        allow_blank=True,
    )
    if rel_start is None:
        rel_start = 0
    if rel_stop is None:
        rel_stop = ep_len
    if rel_start >= rel_stop:
        raise ValueError("Episode slice must satisfy start < stop.")
    return key, Selection(
        mode="episode",
        episode=ep,
        episode_start=rel_start,
        episode_stop=rel_stop,
    )


def resolve_selection_from_args(args: argparse.Namespace) -> Selection:
    if args.index is not None:
        return Selection(mode="index", index=args.index)
    if args.start is not None or args.stop is not None:
        if args.start is None or args.stop is None:
            raise ValueError("Both --start and --stop are required for range mode.")
        return Selection(mode="range", start=args.start, stop=args.stop)
    if args.episode is not None:
        return Selection(
            mode="episode",
            episode=args.episode,
            episode_start=args.episode_start,
            episode_stop=args.episode_stop,
        )
    return Selection(mode="index", index=0)


def read_array(arr, selection: Selection, episode_bounds: list[tuple[int, int]]):
    if selection.mode == "index":
        return arr[selection.index], f"index={selection.index}"

    if selection.mode == "range":
        if selection.start >= selection.stop:
            raise ValueError("Range mode requires start < stop.")
        return arr[selection.start : selection.stop], f"range=[{selection.start}:{selection.stop})"

    if selection.mode == "episode":
        ep = selection.episode
        if ep < 0 or ep >= len(episode_bounds):
            raise ValueError(f"Episode index out of bounds: {ep}")
        ep_abs_start, ep_abs_stop = episode_bounds[ep]
        ep_len = ep_abs_stop - ep_abs_start
        rel_start = 0 if selection.episode_start is None else selection.episode_start
        rel_stop = ep_len if selection.episode_stop is None else selection.episode_stop
        if rel_start < 0 or rel_stop > ep_len or rel_start >= rel_stop:
            raise ValueError(
                f"Invalid episode slice [{rel_start}:{rel_stop}) for episode length {ep_len}."
            )
        abs_start = ep_abs_start + rel_start
        abs_stop = ep_abs_start + rel_stop
        return arr[abs_start:abs_stop], (
            f"episode={ep}, rel=[{rel_start}:{rel_stop}), abs=[{abs_start}:{abs_stop})"
        )

    raise ValueError(f"Unknown selection mode: {selection.mode}")


def print_loaded_data(data, key: str, selection_label: str, preview_items: int) -> None:
    np_data = np.asarray(data)
    print(f"\nLoaded key: {key}")
    print(f"Selection: {selection_label}")
    print(f"Result shape: {np_data.shape}")
    print(f"Result dtype: {np_data.dtype}")
    print(f"min={np_data.min()} max={np_data.max()} mean={np_data.mean()}")

    if np_data.size <= preview_items:
        print("Values:")
        print(np_data)
    else:
        flat = np_data.reshape(-1)
        sample = flat[:preview_items]
        print(f"Preview first {preview_items} flattened values:")
        print(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read UMI-format Zarr dataset arrays with key/episode/timestep selection."
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=str(DEFAULT_ZARR_PATH),
        help="Path to UMI Zarr store.",
    )
    parser.add_argument("--list", action="store_true", help="List available keys and dataset summary.")
    parser.add_argument("--interactive", action="store_true", help="Force interactive key/selection prompts.")
    parser.add_argument("--key", type=str, help="Key under /data to read (example: robot0_eef_pos).")
    parser.add_argument("--index", type=int, help="Read one absolute timestep index.")
    parser.add_argument("--start", type=int, help="Absolute range start (inclusive).")
    parser.add_argument("--stop", type=int, help="Absolute range stop (exclusive).")
    parser.add_argument("--episode", type=int, help="Read from a specific episode index.")
    parser.add_argument("--episode-start", type=int, default=None, help="Episode-relative start (inclusive).")
    parser.add_argument("--episode-stop", type=int, default=None, help="Episode-relative stop (exclusive).")
    parser.add_argument(
        "--preview-items",
        type=int,
        default=20,
        help="How many flattened values to preview when output is large (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.zarr_path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {path}")

    register_umi_codecs()
    root = load_dataset(path)
    keys = print_summary(root)
    episode_ends = np.asarray(root["meta"]["episode_ends"])
    episode_bounds = build_episode_bounds(episode_ends)

    if args.list and not args.interactive and args.key is None:
        return 0

    if args.interactive or args.key is None:
        if not sys.stdin.isatty():
            raise ValueError(
                "Interactive mode requires a TTY. Pass --key and selection flags for non-interactive usage."
            )
        key, selection = interactive_selection(keys, episode_bounds)
    else:
        if args.key not in keys:
            raise ValueError(f"Unknown key: {args.key}. Use --list to see valid keys.")
        key = args.key
        selection = resolve_selection_from_args(args)

    arr = root["data"][key]
    data, selection_label = read_array(arr, selection, episode_bounds)
    print_loaded_data(data, key, selection_label, preview_items=args.preview_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
