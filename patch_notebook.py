import json

path = '/Users/rebnoob/Downloads/Umi Data/notebooks/umi_to_lerobot_v3_retarget_workflow.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'q_sol = chain.inverse_kinematics_frame(T_so_tcp, initial_position=q_full, max_iter=IK_MAX_ITERS)' in line:
                # Find the 'if np.all(np.isfinite(q_sol)):' line
                for j in range(i, len(source)):
                    if 'if np.all(np.isfinite(q_sol)):' in source[j]:
                        print("Found block to replace")
                        # Replace lines
                        source[i] = "        q_sol = chain.inverse_kinematics_frame(T_so_tcp, initial_position=q_full, max_iter=IK_MAX_ITERS)\n"
                        source.insert(i+1, "        T_fk_test = chain.forward_kinematics(q_sol)\n")
                        source.insert(i+2, "        err_test = float(np.linalg.norm(T_fk_test[:3, 3] - T_so_tcp[:3, 3]))\n")
                        source.insert(i+3, "        if err_test > 0.05:\n")
                        source.insert(i+4, "            q_rec = np.zeros(len(chain.links), dtype=np.float64)\n")
                        source.insert(i+5, "            if len(active_joint_indices) >= 3:\n")
                        source.insert(i+6, "                q_rec[active_joint_indices[1]] = -1.0\n")
                        source.insert(i+7, "                q_rec[active_joint_indices[2]] = 1.0\n")
                        source.insert(i+8, "            q_sol2 = chain.inverse_kinematics_frame(T_so_tcp, initial_position=q_rec, max_iter=IK_MAX_ITERS * 2)\n")
                        source.insert(i+9, "            T_fk2 = chain.forward_kinematics(q_sol2)\n")
                        source.insert(i+10, "            err2 = float(np.linalg.norm(T_fk2[:3, 3] - T_so_tcp[:3, 3]))\n")
                        source.insert(i+11, "            if err2 < err_test:\n")
                        source.insert(i+12, "                q_sol = q_sol2\n")
                        
                        break
                break

with open(path, 'w') as f:
    json.dump(nb, f, indent=2)

print("Patched notebook successfully")
