# Writes dev/raytrace/kaggle_render.ipynb, a notebook that ray-traces the block on a free notebook
# service's graphics card. Run: python3 dev/raytrace/kaggle_render.ipynb.py
# The notebook only clones this repository and runs cloud_render.sh, so fixes belong in that script.
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/Machinelearning1954/Cloud-Resources.git"
BRANCH = "ccr-f1f4f089-qjdhbe"
K3D = "/kaggle/working/repo/neon-harbor/prototypes/kingston-3d"
cells = [
    ("markdown",
     "# Neon Harbor: the Kingston block, ray-traced on a free graphics card\n\n"
     "**Before running:** in the notebook's Settings turn **Internet on** and set **Accelerator** to a GPU "
     "(T4 or P100). Then choose **Run all**.\n\n"
     "What it does: fetches the project, installs Blender and a browser into the notebook's own folder "
     "(several minutes, about 1 GB), exports the game's scene from the built page, and renders it with "
     "Blender Cycles on the graphics card. Pictures appear at the bottom and in `/kaggle/working/renders`.\n\n"
     "These are offline pictures of the game's own scene. They are not the game running.\n\n"
     "**Status:** the script this runs was tested end to end on a CPU. It has **not** been run on this "
     "service or on a graphics card yet. If a cell fails, copy the last lines of its output back to the "
     "session that made it."),
    ("code", f"!git clone --depth 1 -b {BRANCH} {REPO} /kaggle/working/repo"),
    ("code", f"!bash {K3D}/dev/raytrace/cloud_render.sh /kaggle/working/renders"),
    ("code",
     "import glob\nfrom IPython.display import Image, display\n"
     "for f in sorted(glob.glob('/kaggle/working/renders/*/*.png')):\n    print(f)\n    display(Image(filename=f, width=360))"),
    ("code", "import shutil\nshutil.make_archive('/kaggle/working/renders', 'zip', '/kaggle/working/renders')\nprint('renders.zip is in the Output panel')"),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(HERE, "kaggle_render.ipynb"), "w") as fh:
    fh.write(json.dumps(nb, indent=1) + "\n")
print("wrote kaggle_render.ipynb")
