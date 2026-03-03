# Umi to LeRobot Retargeting (SO-101)

This repository provides an end-to-end pipeline to convert [UMI](https://umi-gripper.github.io/) (Universal Manipulation Interface) datasets in Zarr format to [LeRobot](https://github.com/huggingface/lerobot) v3 dataset format, specifically targeted for an SO-101 robotic arm.

The core script is `notebooks/umi_to_lerobot_v3_retarget_workflow.ipynb`. It is a 7-step workflow with verifiable checks at each step, ensuring robust and correct conversion.

## Features

- **End-to-End Conversion**: Translates raw UMI Zarr data (`T × features`) into LeRobot's v3 dataset format.
- **FPS Resampling**: Safely converts between differing source framerates (e.g., ~60 FPS from UMI) and target framerates (e.g., 30 FPS).
- **Multiple Retargeting Modes**:
  - `auto`: Starts by trying `joint_passthrough`, falls back to `ik`, and then `proxy` if none are available.
  - `joint_passthrough`: Direct mapping from UMI joint arrays (if available) to the 5 joints + 1 gripper format.
  - `ik` (Inverse Kinematics): Uses the provided SO-101 URDF to calculate real joint angles using IKPy from End-Effector (EEF) poses. Evaluates reachability, continuity (`max |dq|`), and forward kinematics position errors.
  - `proxy`: Fallback that passes raw EEF poses as "joints" for pipeline testing and debugging.
- **Verification Support**: Can validate generated datasets against a "Golden Reference" dataset (recorded with `lerobot-record`) to ensure schema compatibility.

## Directory Structure
- `data/raw/`: Store your raw `.zarr` files here.
- `assets/urdf/`: Contains the URDF models for Inverse Kinematics calculation.
- `notebooks/`: Contains the Jupyter Notebook workflows.
- `tools/maintenance/`: One-off notebook patch/export helpers.
- `outputs/datasets/`: Outputs the final `lerobot` compatible dataset.
- `outputs/videos/`: Output for debug videos.
- `outputs/reports/`: JSON/CSV diagnostics from collision and replay checks.
- `lerobot/`: Expects a local clone of `huggingface/lerobot`.

## Setup & Requirements

1. **Python Environment**: ensure you have a dedicated environment (e.g., conda or venv).
2. **Dependencies**: 
    ```bash
    pip install zarr numcodecs imagecodecs imagecodecs-numcodecs pandas pyarrow matplotlib opencv-python ikpy scipy
    ```
3. **LeRobot**: This notebook expects LeRobot v0.4.4 (v3 dataset codepath). Ensure it's reachable. The notebook is configured to load a local repository via `sys.path` (e.g., `lerobot/src` mapped to your home directory or cloned repository).

## Usage Guide

1. **Configure Paths**: Open `notebooks/umi_to_lerobot_v3_retarget_workflow.ipynb` and locate the `# User configuration` block (Step 0).
   Ensure the following paths match your environment:
   - `UMI_ZARR_PATH`: Path to your raw input `pick_cube.zarr`.
   - `LOCAL_LEROBOT_REPO`: Path to your cloned LeRobot directory.
   - `SO101_URDF_PATH`: Path to the SO-101 URDF file (if using IK).
   - `OUTPUT_ROOT`: The output path for your generated LeRobot dataset.
   - `DEBUG_VIDEO_PATH`: Path to output a debug MP4 video.

2. **Run Pipeline**: Execute the notebook step-by-step.
   - **Step 1:** Loads optional golden dataset for schema enforcement.
   - **Step 2:** Inspects your Zarr array hierarchy and logs information.
   - **Step 3:** Validates 6D (5 joint, 1 gripper) structural definition.
   - **Step 4:** Processes the timeline indices to handle UMI (~60 fps) to Target (30 fps) downsampling.
   - **Step 5:** Executes IK / passthrough generation on every timestamp. It automatically computes gripper states.
   - **Step 6 & 7:** Creates the structure using `LeRobotDataset.add_frame()` and `save_episode()`.

## Output

A standard LeRobot v3 structure will be created inside `OUTPUT_ROOT`. You should expect folders like `meta/`, `videos/`, and raw parquet logs that are fully compatible with HuggingFace `lerobot` training pipelines.

## Run on Real SO-101 (Feetech)

Use:
- `scripts/replay_so101_real.py`

This script converts the retargeted dataset action format (`joint_1..joint_5, gripper`) into SO-101 motor commands (`*.pos`) and handles unit conversion (radians to degrees, gripper 0-1 to 0-100).

### 1) Dry-run (no motor movement)

```bash
python scripts/replay_so101_real.py \
  --dataset-root "outputs/datasets/lerobot_umi_pick_cube_so101_v3" \
  --episode 0
```

### 2) Real replay

```bash
python scripts/replay_so101_real.py \
  --dataset-root "outputs/datasets/lerobot_umi_pick_cube_so101_v3" \
  --episode 0 \
  --port /dev/tty.usbmodem5A460814411 \
  --robot-id so101_real \
  --execute
```

### 3) Useful safety flags

- `--max-relative-target-deg 5`: clamp per-step arm motion to 5 degrees.
- `--max-relative-target-gripper 12`: clamp per-step gripper change.
- `--go-to-start-seconds 2.0`: smoothly move from current pose to first replay frame.
- `--skip-calibrate`: skip automatic calibration prompt (use only if already calibrated).

## Verify Folder Compatibility

Run this after moving the workspace to a new location:

```bash
python scripts/verify_workspace_compatibility.py
```

It validates required directories/files and default paths used by scripts and notebooks.
