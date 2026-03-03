#!/usr/bin/env python3
"""Quick workspace compatibility check after moving/reorganizing folders."""

from __future__ import annotations

from pathlib import Path


def check(path: Path, label: str, must_exist: bool = True) -> bool:
    exists = path.exists()
    status = "PASS" if (exists or not must_exist) else "FAIL"
    print(f"[{status}] {label}: {path}")
    return exists or not must_exist


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ok = True
    ok &= check(root / "data" / "raw", "raw data directory")
    ok &= check(root / "assets" / "urdf", "URDF directory")
    ok &= check(root / "notebooks", "notebooks directory")
    ok &= check(root / "scripts", "scripts directory")
    ok &= check(root / "tools" / "maintenance", "maintenance tools directory")
    ok &= check(root / "outputs" / "datasets", "datasets output directory")
    ok &= check(root / "outputs" / "videos", "videos output directory")
    ok &= check(root / "outputs" / "reports", "reports output directory")
    ok &= check(root / "outputs" / "images", "images output directory")
    ok &= check(root / "lerobot" / "src", "local lerobot source directory")

    # Common default artifacts used by scripts/notebook.
    ok &= check(root / "data" / "raw" / "pick_cube.zarr", "default UMI zarr dataset")
    ok &= check(root / "assets" / "urdf" / "so101_new_calib.urdf", "default SO-101 URDF")
    ok &= check(
        root / "outputs" / "datasets" / "lerobot_umi_pick_cube_so101_v3",
        "default converted LeRobot dataset",
    )

    if not ok:
        print("\nOne or more required paths are missing.")
        return 1

    print("\nWorkspace compatibility check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
