import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "umi_to_lerobot_v3_retarget_workflow.ipynb"

with NOTEBOOK.open("r", encoding="utf-8") as f:
    nb = json.load(f)

replacements = {
    f"{ROOT.as_posix()}/": "",
    (Path.home() / "lerobot").as_posix(): "lerobot",
}

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", [])
    updated = []
    for line in source:
        for old, new in replacements.items():
            line = line.replace(old, new)
        updated.append(line)
    cell["source"] = updated

with NOTEBOOK.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook paths normalized:", NOTEBOOK)
