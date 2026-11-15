# Generates dev/raytrace/kaggle_render.ipynb: a Kaggle notebook that renders the exported scene
# with Blender Cycles on Kaggle's free GPU (CUDA). Run: python3 dev/raytrace/kaggle_render.ipynb.py
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = [
 ("markdown", "# Neon Harbor: Kingston block, ray-traced on a free Kaggle GPU\n\nSettings: Accelerator = GPU T4 x1 (or P100). Internet: on, for the one install line. Upload `scene-gold.json` and `scene-night.json` (from `dev/raytrace/export.js`) as a dataset and add it to this notebook. Renders go to `/kaggle/working`. Offline pictures of the game's own scene, not the game."),
 ("code", "!pip install -q bpy==5.3.0"),
 ("code", "import glob, os\nsrc = glob.glob('/kaggle/input/**/scene-*.json', recursive=True)\nprint(src)\nassert src, 'add the dataset with the scene files first'"),
 ("code", "# Upload dev/raytrace/render.py to /kaggle/working first, then:\nimport subprocess\nfor s in src:\n    name = os.path.basename(s)[:-5]\n    subprocess.run(['python3', '/kaggle/working/render.py', s, f'/kaggle/working/{name}', '--device', 'gpu', '--samples', '256', '--size', '1080x1920', '--views', 'hero,quay,aerial,street'], check=True)"),
 ("code", "import shutil\nshutil.make_archive('/kaggle/working/renders', 'zip', '/kaggle/working')\nprint('done: download /kaggle/working/renders.zip from the Output panel')"),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
open(os.path.join(HERE, "kaggle_render.ipynb"), "w").write(json.dumps(nb, indent=1))
print("wrote kaggle_render.ipynb")
