import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "notebooks" / "umi_to_lerobot_v3_retarget_workflow.ipynb"
with path.open("r", encoding="utf-8") as f:
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
py_out = ROOT / "notebooks" / "umi_to_lerobot_v3_retarget_workflow.py"
with py_out.open("w", encoding="utf-8") as f:
    f.write(final_code)

subprocess.run([os.environ.get("PYTHON", "python"), str(py_out)], check=False)
