import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
z = zarr.open(str(ROOT / "data" / "raw" / "pick_cube.zarr"), mode='r')
ep_ends = z['meta']['episode_ends'][:]
starts = np.concatenate([[0], ep_ends[:-1]])
ep_idx = 1
start, end = starts[ep_idx], ep_ends[ep_idx]
pos = z['data']['robot0_eef_pos'][start:end]

T_so_base_from_umi_base = np.eye(4) # placeholder? The notebook has np.eye(4) too
rot = z['data']['robot0_eef_rot_axis_angle'][start:end]

# let's run IK manually here to see where it fails
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation
chain = Chain.from_urdf_file(str(ROOT / "assets" / "urdf" / "so101_new_calib.urdf"))
active_joint_indices = [i for i, is_active in enumerate(chain.active_links_mask) if is_active]
q_full = np.zeros(len(chain.links), dtype=np.float64)

ok_list = []
z_s = []

for i in range(len(pos)):
    # build rot matrix
    rvec = rot[i]
    if Rotation is not None:
        R = Rotation.from_rotvec(rvec).as_matrix()
    else:
        R = np.eye(3)
        theta = np.linalg.norm(rvec)
        if theta > 1e-12:
            k = rvec / theta
            kx, ky, kz = k
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos[i]
    
    q_sol = chain.inverse_kinematics_frame(T, initial_position=q_full, max_iter=100)
    
    # check if FK of q_sol actually reaches target
    T_fk = chain.forward_kinematics(q_sol)
    pos_err = np.linalg.norm(T_fk[:3, 3] - T[:3, 3])
    
    # IF STUCK, TRY RE-SOLVING FROM ZERO!
    if pos_err > 0.05:
        q_sol_retry = chain.inverse_kinematics_frame(T, initial_position=np.zeros(len(chain.links)), max_iter=200)
        T_fk_retry = chain.forward_kinematics(q_sol_retry)
        pos_err_retry = np.linalg.norm(T_fk_retry[:3, 3] - T[:3, 3])
        if pos_err_retry < pos_err:
            q_sol = q_sol_retry
            pos_err = pos_err_retry
            T_fk = T_fk_retry

    q_full = q_sol
    z_s.append(T_fk[2, 3])
    ok_list.append(pos_err < 0.05)

plt.figure()
plt.plot(pos[:, 2], label='Raw UMI Z')
plt.plot(z_s, label='IK Z', linestyle='dashed')
plt.legend()
out = ROOT / "outputs" / "images" / "debug" / "ep1_ik_z_retry.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))

print("Fails after retry logic:", pos.shape[0] - sum(ok_list))
