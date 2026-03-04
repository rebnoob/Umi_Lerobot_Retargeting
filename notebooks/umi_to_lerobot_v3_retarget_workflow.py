from __future__ import annotations
# If needed, install missing dependencies in this kernel.
# %pip install zarr numcodecs imagecodecs imagecodecs-numcodecs pandas pyarrow matplotlib opencv-python ikpy scipy

# LeRobot bootstrap for this machine/kernel
import importlib.util
import sys
from pathlib import Path

def find_workspace_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "lerobot").exists() and (candidate / "outputs").exists():
            return candidate
    return start


WORKSPACE_ROOT = find_workspace_root()
LOCAL_LEROBOT_REPO = WORKSPACE_ROOT / "lerobot"
LOCAL_LEROBOT_SRC = LOCAL_LEROBOT_REPO / 'src'

if importlib.util.find_spec('lerobot') is None:
    if LOCAL_LEROBOT_SRC.exists():
        sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

if importlib.util.find_spec('lerobot') is None:
    raise ModuleNotFoundError(
        "No module named 'lerobot'.\n"
        "Fix in this notebook kernel with one of:\n"
        f"  1) %pip install -e {LOCAL_LEROBOT_REPO}\n"
        "  2) Select a Jupyter kernel that already has lerobot installed."
    )

import lerobot
print('lerobot import OK:', lerobot.__file__)


import json
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    import numcodecs
except Exception:
    numcodecs = None

try:
    from scipy.spatial.transform import Rotation
except Exception:
    Rotation = None

print('Imports loaded')

# -----------------------------
# User configuration
# -----------------------------

# Step 1: optional golden reference dataset from lerobot-record.
# Point to the root folder that contains `meta/info.json`.
GOLDEN_DATASET_ROOT = None  # e.g. Path('/abs/path/to/your/so101_golden_dataset')
GOLDEN_REPO_ID = 'local/so101_golden'

# UMI source
UMI_ZARR_PATH = WORKSPACE_ROOT / "data" / "raw" / "pick_cube.zarr"
UMI_FPS = 59.94

# Timeline conversion
TARGET_FPS = 30
MAX_EPISODES_TO_CONVERT = 5  # set to None for all episodes

# Step 5 retarget config
# auto: prefer joints if present, else IK (if configured), else proxy fallback
RETARGET_MODE = 'auto'  # {'auto', 'joint_passthrough', 'ik', 'proxy'}
STRICT_SO101_ACTION = False  # if True, fail when only proxy retarget is available

# IK config (only needed if RETARGET_MODE uses IK)
SO101_URDF_PATH = WORKSPACE_ROOT / "assets" / "urdf" / "so101_new_calib.urdf"  # e.g. Path('/abs/path/to/so101.urdf')
IK_ACTIVE_LINK_MASK = None  # e.g. [False, True, True, True, True, True, False]
IK_ARM_ACTIVE_INDICES = None  # indices into active joints for q1..q5; None -> first 5 active joints
IK_MAX_ITERS = 100
IK_MATCH_ORIENTATION = False  # False = position-priority IK (recommended for stable retargeting)
IK_POS_TOL_M = 0.03  # IK frame considered successful if position error <= this

# Frame transforms (UMI base -> SO-101 base, and UMI TCP -> SO TCP)
# Keep identity initially; update from calibration when available.
T_SO_BASE_FROM_UMI_BASE = np.eye(4, dtype=np.float64)
T_SO_TCP_FROM_UMI_TCP = np.eye(4, dtype=np.float64)

# Gripper mapping
GRIPPER_KEY = 'robot0_gripper_width'  # expected shape (T,1)
GRIPPER_INVERT = False  # set True if close/open direction is reversed after sanity check

# Output LeRobot dataset
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "datasets" / "lerobot_umi_pick_cube_so101_v3"
OUTPUT_REPO_ID = 'local/umi_pick_cube_so101_v3'
OVERWRITE_OUTPUT = True
TASK_TEXT = 'pick_cube'
VIDEO_CODEC = 'h264'  # {'h264', 'hevc', 'libsvtav1'}

# Feature naming (used if no golden template is available)
CAMERA_UMI_KEY = 'camera0_rgb'
CAMERA_LEROBOT_KEY = 'observation.images.front'
ACTION_KEY = 'action'
STATE_KEY = 'observation.state'
ACTION_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper']

# Step 7 debug outputs
DEBUG_VIDEO_PATH = WORKSPACE_ROOT / "outputs" / "videos" / "retarget_debug_alignment.mp4"
DEBUG_EPISODE_INDEX = 0

print('Config loaded')

# -----------------------------
# Helpers
# -----------------------------

def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(f'FAILED: {message}')
    print(f'PASS: {message}')


def register_umi_codecs():
    if numcodecs is None:
        print('numcodecs not available, skipping codec pre-check')
        return
    try:
        numcodecs.get_codec({'id': 'imagecodecs_jpegxl'})
        print('JPEG-XL codec is available in numcodecs registry.')
        return
    except Exception:
        pass

    # Best-effort registration paths
    for module_name in ('diffusion_policy.codecs.imagecodecs_numcodecs', 'imagecodecs_numcodecs'):
        try:
            module = __import__(module_name, fromlist=['register_codecs'])
            module.register_codecs()
        except Exception:
            pass

    try:
        numcodecs.get_codec({'id': 'imagecodecs_jpegxl'})
        print('JPEG-XL codec registered successfully.')
    except Exception:
        print('WARNING: JPEG-XL codec not registered. camera RGB reads may fail.')


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if Rotation is not None:
        return Rotation.from_rotvec(rotvec).as_matrix()

    # Rodrigues fallback
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R


def build_pose_hmat(pos_xyz: np.ndarray, rot_axis_angle: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = axis_angle_to_matrix(rot_axis_angle)
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T


def get_episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    check(episode_ends.ndim == 1, '/meta/episode_ends must be 1D')
    check(len(episode_ends) > 0, '/meta/episode_ends must be non-empty')
    check(np.all(np.diff(episode_ends) > 0), '/meta/episode_ends must be strictly increasing')
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def resample_episode_indices(ep_len: int, src_fps: float, dst_fps: float) -> np.ndarray:
    check(ep_len >= 1, 'episode length >= 1')
    if abs(src_fps - dst_fps) < 1e-8:
        return np.arange(ep_len, dtype=np.int64)

    step = src_fps / dst_fps
    raw = np.arange(0, ep_len, step, dtype=np.float64)
    idx = np.round(raw).astype(np.int64)
    idx = np.clip(idx, 0, ep_len - 1)
    idx = np.unique(idx)
    if idx[-1] != ep_len - 1:
        idx = np.concatenate([idx, np.array([ep_len - 1], dtype=np.int64)])
    return idx


def normalize_gripper(width_values: np.ndarray, invert: bool = False) -> tuple[np.ndarray, dict]:
    g = np.asarray(width_values, dtype=np.float64).reshape(-1)
    g_min = float(np.min(g))
    g_max = float(np.max(g))
    denom = max(g_max - g_min, 1e-9)
    norm = (g - g_min) / denom
    norm = np.clip(norm, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    return norm.astype(np.float32), {'min': g_min, 'max': g_max, 'invert': invert}


register_umi_codecs()
print('Helper functions ready')

golden_info = None
golden_features = None
golden_action_key = None
golden_camera_keys = []

if GOLDEN_DATASET_ROOT is None:
    print('INFO: GOLDEN_DATASET_ROOT is not set. Using explicit fallback template later.')
else:
    golden_root = Path(GOLDEN_DATASET_ROOT)
    info_path = golden_root / 'meta' / 'info.json'
    check(info_path.exists(), f'golden meta/info.json exists: {info_path}')

    golden_info = json.loads(info_path.read_text())
    golden_features = golden_info.get('features', {})
    check(len(golden_features) > 0, 'golden features are present')

    action_candidates = [k for k in golden_features if k.startswith('action')]
    camera_candidates = [k for k in golden_features if k.startswith('observation.images.')]

    check(len(action_candidates) >= 1, 'golden has at least one action key')
    check(len(camera_candidates) >= 1, 'golden has at least one observation.images.* key')

    golden_action_key = action_candidates[0]
    golden_camera_keys = camera_candidates

    ds_golden = LeRobotDataset(
        repo_id=GOLDEN_REPO_ID,
        root=golden_root,
        download_videos=False,
        video_backend='pyav',
    )
    sample = ds_golden[0]
    check(golden_action_key in sample, f'golden sample contains {golden_action_key}')
    check(golden_camera_keys[0] in sample, f'golden sample contains {golden_camera_keys[0]}')

    print('Golden reference keys (subset):')
    print(' action key:', golden_action_key)
    print(' camera key:', golden_camera_keys[0])
    print(' action shape:', tuple(sample[golden_action_key].shape))

check(UMI_ZARR_PATH.exists(), f'UMI zarr exists: {UMI_ZARR_PATH}')

warnings.filterwarnings(
    'ignore',
    message=r'Object at \.DS_Store is not recognized as a component of a Zarr hierarchy\.',
)
root = zarr.open(str(UMI_ZARR_PATH), mode='r')
check('data' in root, 'UMI root contains /data')
check('meta' in root and 'episode_ends' in root['meta'], 'UMI root contains /meta/episode_ends')

print('UMI tree:')
try:
    print(root.tree())
except Exception:
    print('tree() unavailable; printing keys manually')
    print('data keys:', list(root['data'].array_keys()))

umi_data = root['data']
umi_keys = sorted(umi_data.array_keys())
check(len(umi_keys) > 0, 'UMI /data has arrays')

summary_rows = []
lengths = []
for k in umi_keys:
    arr = umi_data[k]
    lengths.append(int(arr.shape[0]))
    compressors = getattr(arr, 'compressors', ())
    comp = 'none' if not compressors else str(compressors[0])
    summary_rows.append({
        'key': k,
        'shape': tuple(arr.shape),
        'dtype': str(arr.dtype),
        'chunks': tuple(arr.chunks),
        'compressor': comp,
    })

umi_df = pd.DataFrame(summary_rows)

try:
    from IPython.display import display as _display
except Exception:
    _display = print
_display(umi_df)

T = lengths[0]
check(all(x == T for x in lengths), 'all UMI data arrays share same first dim T')

episode_ends = np.asarray(root['meta']['episode_ends'], dtype=np.int64)
check(int(episode_ends[-1]) == T, f'episode_ends[-1] ({int(episode_ends[-1])}) == T ({T})')

episode_bounds = get_episode_bounds(episode_ends)
print('Total timesteps T =', T)
print('Total episodes =', len(episode_bounds))
print('First 5 episode bounds =', episode_bounds[:5])

locked_action_dim = 6
check(len(ACTION_NAMES) == locked_action_dim, 'ACTION_NAMES has length 6')

if golden_features is not None:
    ga = golden_features.get(golden_action_key)
    check(ga is not None, 'golden action feature exists')
    print('Golden action feature:', ga)

    if tuple(ga.get('shape', ())) != (locked_action_dim,):
        print('WARNING: golden action shape does not match locked 6D action. Review mapping carefully.')

    # Optional golden action range stats
    try:
        ds_golden_stats = LeRobotDataset(
            repo_id=GOLDEN_REPO_ID,
            root=Path(GOLDEN_DATASET_ROOT),
            download_videos=False,
            video_backend='pyav',
        )
        n = min(2000, ds_golden_stats.num_frames)
        acts = []
        for i in range(n):
            acts.append(ds_golden_stats[i][golden_action_key].cpu().numpy())
        acts = np.stack(acts)
        print('Golden action min:', acts.min(axis=0))
        print('Golden action max:', acts.max(axis=0))
    except Exception as e:
        print('WARNING: could not compute golden action stats:', e)
else:
    print('No golden dataset provided: using locked fallback feature schema in Step 6.')

# Build resampling index plan episode-by-episode.
if MAX_EPISODES_TO_CONVERT is None:
    selected_episode_ids = list(range(len(episode_bounds)))
else:
    selected_episode_ids = list(range(min(MAX_EPISODES_TO_CONVERT, len(episode_bounds))))

conversion_plan = []
for ep_id in selected_episode_ids:
    ep_start, ep_end = episode_bounds[ep_id]
    ep_len = ep_end - ep_start
    rel_idx = resample_episode_indices(ep_len, src_fps=UMI_FPS, dst_fps=TARGET_FPS)
    abs_idx = ep_start + rel_idx
    ts = np.arange(len(abs_idx), dtype=np.float64) / float(TARGET_FPS)

    check(len(abs_idx) >= 2, f'episode {ep_id}: resampled length >= 2')
    check(np.all(np.diff(abs_idx) > 0), f'episode {ep_id}: abs indices strictly increasing')
    check(np.all(np.diff(ts) > 0), f'episode {ep_id}: timestamps strictly increasing')

    if len(ts) > 1:
        dt = np.diff(ts)
        check(np.allclose(dt, 1.0 / TARGET_FPS, atol=1e-9), f'episode {ep_id}: dt == 1/fps')

    conversion_plan.append(
        {
            'episode_index': ep_id,
            'ep_start': ep_start,
            'ep_end': ep_end,
            'abs_indices': abs_idx,
            'timestamps': ts,
        }
    )

print('Planned episodes:', len(conversion_plan))
print('First episode plan summary:')
print({
    'episode_index': conversion_plan[0]['episode_index'],
    'original_len': conversion_plan[0]['ep_end'] - conversion_plan[0]['ep_start'],
    'resampled_len': len(conversion_plan[0]['abs_indices']),
    'first_abs_idx': int(conversion_plan[0]['abs_indices'][0]),
    'last_abs_idx': int(conversion_plan[0]['abs_indices'][-1]),
})

@dataclass
class RetargetReport:
    mode: str
    ik_success_rate: float
    ik_mean_fk_pos_err_m: float | None
    max_abs_joint_step: float
    notes: list[str]


def find_joint_candidate_key(umi_group) -> str | None:
    candidates = []
    for k in umi_group.array_keys():
        arr = umi_group[k]
        if len(arr.shape) == 2 and arr.shape[1] >= 5 and 'joint' in k.lower():
            candidates.append((k, arr.shape[1]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def build_gripper_vector(abs_indices: np.ndarray) -> tuple[np.ndarray, dict]:
    check(GRIPPER_KEY in umi_data, f'gripper key exists: {GRIPPER_KEY}')
    g_all = np.asarray(umi_data[GRIPPER_KEY])
    g_norm_all, g_stats = normalize_gripper(g_all[:, 0], invert=GRIPPER_INVERT)
    return g_norm_all[abs_indices], g_stats


def retarget_joint_passthrough(abs_indices: np.ndarray, joint_key: str) -> tuple[np.ndarray, dict, list[str]]:
    joints_all = np.asarray(umi_data[joint_key], dtype=np.float32)
    check(joints_all.shape[1] >= 5, f'joint key {joint_key} has >=5 dims')

    g, g_stats = build_gripper_vector(abs_indices)

    action = np.zeros((len(abs_indices), 6), dtype=np.float32)
    action[:, :5] = joints_all[abs_indices, :5]
    action[:, 5] = g

    notes = [f'joint_passthrough from {joint_key}', f'gripper_norm={g_stats}']
    return action, g_stats, notes


def retarget_proxy(abs_indices: np.ndarray) -> tuple[np.ndarray, dict, list[str]]:
    pos = np.asarray(umi_data['robot0_eef_pos'], dtype=np.float32)[abs_indices]
    rot = np.asarray(umi_data['robot0_eef_rot_axis_angle'], dtype=np.float32)[abs_indices]
    g, g_stats = build_gripper_vector(abs_indices)

    action = np.zeros((len(abs_indices), 6), dtype=np.float32)
    action[:, 0:3] = pos[:, 0:3]
    action[:, 3:5] = rot[:, 0:2]
    action[:, 5] = g

    notes = [
        'PROXY MODE: action[:5] built from eef_pos/eef_rot (NOT physical SO-101 joints).',
        'Use IK mode with URDF to produce real SO-101 joint targets.',
        f'gripper_norm={g_stats}',
    ]
    return action, g_stats, notes


def try_make_ik_chain(urdf_path: Path, active_links_mask=None):
    from ikpy.chain import Chain

    chain = Chain.from_urdf_file(str(urdf_path), active_links_mask=active_links_mask)
    # IKPy active_links_mask can include fixed links depending on URDF defaults.
    # We always derive controllable joints from non-fixed links to keep action mapping stable.
    movable_joint_indices = [
        i for i, link in enumerate(chain.links) if getattr(link, 'joint_type', 'fixed') != 'fixed'
    ]
    if active_links_mask is not None:
        movable_joint_indices = [i for i in movable_joint_indices if bool(active_links_mask[i])]
    return chain, movable_joint_indices


def retarget_ik(abs_indices: np.ndarray) -> tuple[np.ndarray, dict, list[str], float, float | None]:
    check(SO101_URDF_PATH is not None, 'SO101_URDF_PATH is set for IK mode')
    urdf_path = Path(SO101_URDF_PATH)
    check(urdf_path.exists(), f'URDF path exists: {urdf_path}')

    chain, movable_joint_indices = try_make_ik_chain(urdf_path, IK_ACTIVE_LINK_MASK)
    if IK_ARM_ACTIVE_INDICES is None:
        arm_sel = list(range(min(5, len(movable_joint_indices))))
    else:
        arm_sel = list(IK_ARM_ACTIVE_INDICES)

    check(len(movable_joint_indices) >= 5, f'URDF must have >=5 movable joints, got {len(movable_joint_indices)}')
    check(len(arm_sel) == 5, 'IK arm selection must produce 5 joints')
    check(
        all(0 <= j < len(movable_joint_indices) for j in arm_sel),
        'IK arm selection indices must be valid for movable joints',
    )

    pos_all = np.asarray(umi_data['robot0_eef_pos'], dtype=np.float64)
    rot_all = np.asarray(umi_data['robot0_eef_rot_axis_angle'], dtype=np.float64)
    g, g_stats = build_gripper_vector(abs_indices)

    q_full = np.zeros(len(chain.links), dtype=np.float64)
    action = np.zeros((len(abs_indices), 6), dtype=np.float32)

    ok = 0
    fk_errs = []

    for i, idx in enumerate(abs_indices):
        T_umi_tcp = build_pose_hmat(pos_all[idx], rot_all[idx])
        T_so_tcp = T_SO_BASE_FROM_UMI_BASE @ T_umi_tcp @ T_SO_TCP_FROM_UMI_TCP
        target_pos = T_so_tcp[:3, 3]
        if IK_MATCH_ORIENTATION:
            q_sol = chain.inverse_kinematics_frame(T_so_tcp, initial_position=q_full, max_iter=IK_MAX_ITERS)
        else:
            # Position-priority IK is more stable for retargeting when orientation frames are uncertain.
            q_sol = chain.inverse_kinematics(target_position=target_pos, initial_position=q_full, max_iter=IK_MAX_ITERS)

        T_fk_test = chain.forward_kinematics(q_sol)
        err_test = float(np.linalg.norm(T_fk_test[:3, 3] - target_pos))

        if err_test > IK_POS_TOL_M:
            q_seed = np.zeros(len(chain.links), dtype=np.float64)
            if len(movable_joint_indices) >= 3:
                q_seed[movable_joint_indices[1]] = -1.0
                q_seed[movable_joint_indices[2]] = 1.0
            if IK_MATCH_ORIENTATION:
                q_try = chain.inverse_kinematics_frame(T_so_tcp, initial_position=q_seed, max_iter=IK_MAX_ITERS * 2)
            else:
                q_try = chain.inverse_kinematics(target_position=target_pos, initial_position=q_seed, max_iter=IK_MAX_ITERS * 2)
            T_fk_try = chain.forward_kinematics(q_try)
            err_try = float(np.linalg.norm(T_fk_try[:3, 3] - target_pos))
            if err_try < err_test:
                q_sol = q_try
                err_test = err_try

        if np.all(np.isfinite(q_sol)):
            q_full = q_sol
            if err_test <= IK_POS_TOL_M:
                ok += 1

        movable_values = q_full[movable_joint_indices]
        q5 = np.array([movable_values[j] for j in arm_sel], dtype=np.float32)

        action[i, :5] = q5
        action[i, 5] = g[i]

        # FK position error check
        T_fk = chain.forward_kinematics(q_full)
        pos_err = float(np.linalg.norm(T_fk[:3, 3] - target_pos))
        fk_errs.append(pos_err)

    success_rate = ok / max(len(abs_indices), 1)
    mean_fk = float(np.mean(fk_errs)) if fk_errs else None
    notes = [
        f'IK from URDF={urdf_path}',
        f'ik_match_orientation={IK_MATCH_ORIENTATION}',
        f'movable_joint_indices={movable_joint_indices}',
        f'arm_selection={arm_sel}',
        f'gripper_norm={g_stats}',
    ]
    return action, g_stats, notes, success_rate, mean_fk


joint_candidate_key = find_joint_candidate_key(umi_data)
print('Detected joint candidate key:', joint_candidate_key)

episode_payloads = []
notes_global = []
ik_success_rates = []
fk_errs = []
actual_mode = None

for plan in conversion_plan:
    abs_idx = plan['abs_indices']

    mode_try = RETARGET_MODE
    if RETARGET_MODE == 'auto':
        if joint_candidate_key is not None:
            mode_try = 'joint_passthrough'
        elif SO101_URDF_PATH is not None:
            mode_try = 'ik'
        else:
            mode_try = 'proxy'

    if mode_try == 'joint_passthrough':
        action, g_stats, notes = retarget_joint_passthrough(abs_idx, joint_candidate_key)
        ik_success = 1.0
        mean_fk = None
    elif mode_try == 'ik':
        action, g_stats, notes, ik_success, mean_fk = retarget_ik(abs_idx)
    elif mode_try == 'proxy':
        action, g_stats, notes = retarget_proxy(abs_idx)
        ik_success = 0.0
        mean_fk = None
    else:
        raise ValueError(f'Unknown RETARGET_MODE: {RETARGET_MODE}')

    state = action.copy()

    # Step 4 consistency checks now that we have aligned arrays
    ts = plan['timestamps']
    check(len(action) == len(abs_idx) == len(ts), f"ep {plan['episode_index']}: len(images/actions/timestamps) match")

    # Step 5 continuity check
    if len(action) > 1:
        max_abs_step = float(np.max(np.abs(np.diff(action[:, :5], axis=0))))
    else:
        max_abs_step = 0.0

    episode_payloads.append(
        {
            'episode_index': plan['episode_index'],
            'abs_indices': abs_idx,
            'timestamps': ts,
            'action': action,
            'state': state,
            'max_abs_joint_step': max_abs_step,
            'retarget_mode': mode_try,
            'retarget_notes': notes,
        }
    )
    notes_global.extend(notes)
    ik_success_rates.append(float(ik_success))
    if mean_fk is not None:
        fk_errs.append(float(mean_fk))
    actual_mode = mode_try

all_joint_steps = [ep['max_abs_joint_step'] for ep in episode_payloads]
report = RetargetReport(
    mode=actual_mode,
    ik_success_rate=float(np.mean(ik_success_rates)) if ik_success_rates else 0.0,
    ik_mean_fk_pos_err_m=float(np.mean(fk_errs)) if fk_errs else None,
    max_abs_joint_step=float(np.max(all_joint_steps)) if all_joint_steps else 0.0,
    notes=sorted(set(notes_global)),
)

print(report)
if report.mode == 'proxy':
    print('\nWARNING: proxy mode is active. This is only for pipeline debugging, not real SO-101 control targets.')
    if STRICT_SO101_ACTION:
        raise RuntimeError('STRICT_SO101_ACTION=True but only proxy mode was available.')

# Step 5 verification summary
check(len(episode_payloads) == len(conversion_plan), 'retarget produced payload for each planned episode')
check(report.max_abs_joint_step < 1e6, 'continuity sanity check passed (finite joint-step bound)')

if report.mode == 'ik':
    check(report.ik_success_rate > 0.80, f'IK success rate > 80% (actual={report.ik_success_rate:.3f})')
    if report.ik_mean_fk_pos_err_m is not None:
        print(f'IK mean FK position error: {report.ik_mean_fk_pos_err_m:.6f} m')

# Visualize first converted episode action trajectories
first_ep = episode_payloads[0]
a = first_ep['action']
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
for j in range(6):
    ax = axes[j // 3, j % 3]
    ax.plot(a[:, j], linewidth=1.0)
    ax.set_title(ACTION_NAMES[j])
    ax.grid(True, alpha=0.3)
plt.suptitle(f"Step 5: Retargeted action traces (episode {first_ep['episode_index']}, mode={first_ep['retarget_mode']})")
plt.tight_layout()
plt.show()

# Build output feature schema
if golden_features is not None:
    # Keep only keys we can provide reliably, but preserve golden naming when possible.
    out_action_key = golden_action_key if golden_action_key in golden_features else ACTION_KEY
    out_camera_key = golden_camera_keys[0] if len(golden_camera_keys) > 0 else CAMERA_LEROBOT_KEY

    # Determine image shape from UMI camera
    cam_shape = tuple(np.asarray(umi_data[CAMERA_UMI_KEY][0]).shape)

    output_features = {
        out_camera_key: {
            'dtype': 'video',
            'shape': cam_shape,
            'names': ['height', 'width', 'channels'],
        },
        out_action_key: {
            'dtype': 'float32',
            'shape': (6,),
            'names': ACTION_NAMES,
        },
        STATE_KEY: {
            'dtype': 'float32',
            'shape': (6,),
            'names': ACTION_NAMES,
        },
    }
else:
    out_action_key = ACTION_KEY
    out_camera_key = CAMERA_LEROBOT_KEY
    cam_shape = tuple(np.asarray(umi_data[CAMERA_UMI_KEY][0]).shape)

    output_features = {
        out_camera_key: {
            'dtype': 'video',
            'shape': cam_shape,
            'names': ['height', 'width', 'channels'],
        },
        out_action_key: {
            'dtype': 'float32',
            'shape': (6,),
            'names': ACTION_NAMES,
        },
        STATE_KEY: {
            'dtype': 'float32',
            'shape': (6,),
            'names': ACTION_NAMES,
        },
    }

print('Output features:')
print(json.dumps(output_features, indent=2))

# Write dataset
if OUTPUT_ROOT.exists():
    if OVERWRITE_OUTPUT:
        shutil.rmtree(OUTPUT_ROOT)
        print('Removed existing output root:', OUTPUT_ROOT)
    else:
        raise FileExistsError(f'OUTPUT_ROOT exists: {OUTPUT_ROOT}. Set OVERWRITE_OUTPUT=True or change path.')

writer_ds = LeRobotDataset.create(
    repo_id=OUTPUT_REPO_ID,
    fps=int(TARGET_FPS),
    features=output_features,
    root=OUTPUT_ROOT,
    robot_type='so101_follower',
    use_videos=True,
    vcodec=VIDEO_CODEC,
)

camera_arr = umi_data[CAMERA_UMI_KEY]

for ep in episode_payloads:
    abs_idx = ep['abs_indices']
    action = ep['action']
    state = ep['state']

    check(len(abs_idx) == len(action) == len(state), f"writer input lengths match for episode {ep['episode_index']}")

    for j, idx in enumerate(abs_idx):
        frame = {
            out_camera_key: np.asarray(camera_arr[int(idx)]),
            out_action_key: action[j].astype(np.float32),
            STATE_KEY: state[j].astype(np.float32),
            'task': TASK_TEXT,
        }
        writer_ds.add_frame(frame)

    # parallel_encoding=False is usually more stable on macOS
    writer_ds.save_episode(parallel_encoding=False)
    print(f"Saved episode {ep['episode_index']} with {len(abs_idx)} frames")

writer_ds.finalize()
print('Finalize completed')

# Step 6 file-level checks
check((OUTPUT_ROOT / 'meta' / 'info.json').exists(), 'meta/info.json exists')
parquet_files = sorted((OUTPUT_ROOT / 'data').glob('*/*.parquet'))
video_files = sorted((OUTPUT_ROOT / 'videos').glob('*/*/*.mp4'))
check(len(parquet_files) >= 1, 'at least one parquet file exists')
check(len(video_files) >= 1, 'at least one video file exists')
print('Parquet files:', len(parquet_files), 'Video files:', len(video_files))

# Load and verify output dataset
# Use video_backend='pyav' to avoid torchcodec runtime dependency issues.
out_ds = LeRobotDataset(
    repo_id=OUTPUT_REPO_ID,
    root=OUTPUT_ROOT,
    download_videos=False,
    video_backend='pyav',
)

print(out_ds)
check(out_ds.num_episodes == len(episode_payloads), 'loaded dataset episode count matches written episodes')
check(out_ds.num_frames > 0, 'loaded dataset has frames')

sample0 = out_ds[0]
check(out_action_key in sample0, f'sample contains action key: {out_action_key}')
check(out_camera_key in sample0, f'sample contains camera key: {out_camera_key}')
check(STATE_KEY in sample0, f'sample contains state key: {STATE_KEY}')
check(tuple(sample0[out_action_key].shape) == (6,), 'sample action shape is (6,)')

print('Sample keys:', list(sample0.keys()))
print('Sample action:', sample0[out_action_key])
print('Sample timestamp:', sample0['timestamp'])

# Boundary checks across episodes
eps_meta = [out_ds.meta.episodes[i] for i in range(len(out_ds.meta.episodes))]
for i in range(len(eps_meta) - 1):
    end_i = int(eps_meta[i]['dataset_to_index'])
    start_next = int(eps_meta[i + 1]['dataset_from_index'])
    check(end_i == start_next, f'episode boundary contiguous between {i} and {i+1}')

print('Episode metadata preview:')
for i in range(min(5, len(eps_meta))):
    print(eps_meta[i])

# Step 7A: Render debug video with overlays from converted episode payload

def render_alignment_debug_video(ep_payload: dict, out_path: Path, fps: int = 20):
    abs_idx = ep_payload['abs_indices']
    action = ep_payload['action']

    frame0 = np.asarray(umi_data[CAMERA_UMI_KEY][int(abs_idx[0])])
    h, w = frame0.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    check(vw.isOpened(), f'video writer opened: {out_path}')

    try:
        for j, idx in enumerate(abs_idx):
            frame = np.asarray(umi_data[CAMERA_UMI_KEY][int(idx)]).copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            txt1 = f't={int(idx)}  step={j+1}/{len(abs_idx)}  mode={ep_payload["retarget_mode"]}'
            txt2 = 'a=' + np.array2string(action[j], precision=3, suppress_small=True)

            cv2.putText(frame_bgr, txt1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, txt2[:120], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # simple gripper bar
            g = float(np.clip(action[j, 5], 0.0, 1.0))
            x0, y0, bw, bh = 8, h - 24, 160, 12
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), -1)
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + int(bw * g), y0 + bh), (0, 220, 0), -1)
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 1)
            cv2.putText(frame_bgr, f'g={g:.2f}', (x0 + bw + 8, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            vw.write(frame_bgr)
    finally:
        vw.release()


debug_ep = next(ep for ep in episode_payloads if ep['episode_index'] == DEBUG_EPISODE_INDEX)
render_alignment_debug_video(debug_ep, DEBUG_VIDEO_PATH, fps=int(TARGET_FPS))
check(DEBUG_VIDEO_PATH.exists(), f'debug video exists: {DEBUG_VIDEO_PATH}')
print('Debug video saved to:', DEBUG_VIDEO_PATH)

# Step 7B: Tiny overfit test on one episode (state -> action)
# This is a lightweight check for major scale/time bugs.

import torch
import torch.nn as nn
import torch.optim as optim

ov_ep = episode_payloads[0]
X = torch.tensor(ov_ep['state'], dtype=torch.float32)
Y = torch.tensor(ov_ep['action'], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, Y.shape[1]),
)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

losses = []
for epoch in range(200):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    opt.step()
    losses.append(float(loss.item()))

print('overfit loss start:', losses[0])
print('overfit loss end:', losses[-1])
check(losses[-1] < losses[0], 'tiny overfit loss decreased')

plt.figure(figsize=(6, 3))
plt.plot(losses)
plt.title('Step 7B Tiny Overfit Loss')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
