"""Write COLMAP .bin camera files next to the .txt ones the dataset ships.

The LiveSplats loader only reads cameras/{cameras,images}.bin.
usage: python txt2bin.py <data_root>   (converts every cameras/ dir below it)
"""
import os
import sys

sys.path.insert(0, "/scratch/hbommaka/code/basket-multiview/livesplats")
from utils.colmap_utils import read_model, write_model  # noqa: E402

for root, dirs, files in os.walk(sys.argv[1]):
    if os.path.basename(root) == "cameras" and "images.txt" in files:
        if os.path.exists(os.path.join(root, "images.bin")):
            continue
        cams, imgs, pts = read_model(root, ext=".txt")
        write_model(cams, imgs, pts, root, ext=".bin")
        print("converted %s (%d cameras)" % (root, len(cams)))
