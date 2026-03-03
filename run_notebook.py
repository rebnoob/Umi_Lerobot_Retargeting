import json

path = '/Users/rebnoob/Downloads/Umi Data/notebooks/umi_to_lerobot_v3_retarget_workflow.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

lines = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for line in cell['source']:
            lines.append(line)
        lines.append('\n')

# move from __future__ to top
out_lines = []
future_lines = []
for line in lines:
    if 'from __future__ import' in line:
        future_lines.append(line)
    else:
        out_lines.append(line)

final_code = "".join(future_lines + out_lines)
with open('notebooks/umi_to_lerobot_v3_retarget_workflow.py', 'w') as f:
    f.write(final_code)

import os
os.system('/opt/anaconda3/envs/umi_lerobot/bin/python notebooks/umi_to_lerobot_v3_retarget_workflow.py')
