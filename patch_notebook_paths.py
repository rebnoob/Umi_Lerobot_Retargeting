import json

path = '/Users/rebnoob/Downloads/Umi Data/notebooks/umi_to_lerobot_v3_retarget_workflow.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "LOCAL_LEROBOT_REPO = Path('/Users/rebnoob/lerobot')" in line:
                source[i] = line.replace("/Users/rebnoob/lerobot", "/Users/rebnoob/Downloads/Umi Data/lerobot")
            elif "UMI_ZARR_PATH = Path('/Users/rebnoob/Downloads/Umi Data/pick_cube.zarr')" in line:
                source[i] = line.replace("pick_cube.zarr", "data/raw/pick_cube.zarr")
            elif "SO101_URDF_PATH = ('/Users/rebnoob/Downloads/Umi Data/urdf/so101_new_calib.urdf')" in line:
                source[i] = line.replace("urdf/so101", "assets/urdf/so101")
            elif "OUTPUT_ROOT = Path('/Users/rebnoob/Downloads/Umi Data/lerobot_umi_pick_cube_so101_v3')" in line:
                source[i] = line.replace("lerobot_umi_pick", "outputs/datasets/lerobot_umi_pick")
            elif "DEBUG_VIDEO_PATH = Path('/Users/rebnoob/Downloads/Umi Data/retarget_debug_alignment.mp4')" in line:
                source[i] = line.replace("retarget_debug", "outputs/videos/retarget_debug")

with open(path, 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook paths patched!")
