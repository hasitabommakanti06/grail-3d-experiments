"""Collect every input the experiments used into one folder (then zip it).

usage: python pack_inputs.py <scratch root> <out dir>
"""
import glob
import json
import os
import shutil
import sys

root, out = sys.argv[1], sys.argv[2]
data, output = os.path.join(root, "data"), os.path.join(root, "output")


def copy_frames(src_masks, src_rgb, src_cams, dst, frames):
    shutil.copytree(src_cams, os.path.join(dst, "cameras"), dirs_exist_ok=True)
    for kind, src in (("masks", src_masks), ("rgb", src_rgb)):
        for cam in sorted(os.listdir(src)):
            for t in frames:
                f = os.path.join(src, cam, "%04d.png" % t)
                if os.path.exists(f):
                    os.makedirs(os.path.join(dst, kind, cam), exist_ok=True)
                    shutil.copy2(f, os.path.join(dst, kind, cam))


# 1. raw frames from the public dataset (only the frames the runs touched)
ap = os.path.join(data, "assets/players_A_pose/male")
copy_frames(os.path.join(ap, "masks"), os.path.join(ap, "rgb/clothing_black/skin_color_1"), os.path.join(ap, "cameras"),
            os.path.join(out, "raw_data/standing_player"), [0])
used = {"running_black": set(), "dribbling_black": set()}
for v in glob.glob(os.path.join(output, "*", "*", "*", "variant.json")) + glob.glob(os.path.join(output, "*", "*", "*", "scene", "variant.json")) \
        + glob.glob(os.path.join(output, "*", "*", "variant.json")):
    j = json.load(open(v))
    if "t0" in j:
        seq = "dribbling_black" if "dribbling_black" in j["seq"] else "running_black"
        used[seq] |= {j["t0"], j["t0"] + j.get("offset", 0)}
for seq, frames in used.items():
    s = os.path.join(data, "development", seq)
    copy_frames(os.path.join(s, "male/masks"), os.path.join(s, "male/rgb/skin_color_1"), os.path.join(s, "cameras"),
                os.path.join(out, "raw_data", "jogging_player" if seq == "running_black" else "dribbling_player"), sorted(frames))
    print(seq, "frames", sorted(frames))

# 2. camera split and every run's scene (cameras actually used, changed masks, settings)
os.makedirs(os.path.join(out, "scenes"), exist_ok=True)
shutil.copy2(os.path.join(output, "exp_camera_count/splits.json"), os.path.join(out, "camera_split.json"))
n = 0
for cams in glob.glob(os.path.join(output, "**", "cameras"), recursive=True):
    scene = os.path.dirname(cams)
    rel = os.path.relpath(scene, output)
    rel = rel[:-len("/scene")] if rel.endswith("/scene") else rel
    dst = os.path.join(out, "scenes", rel)
    shutil.copytree(cams, os.path.join(dst, "cameras"), dirs_exist_ok=True)
    for extra in ("variant.json", "hull.obj"):
        for cand in (os.path.join(scene, extra), os.path.join(os.path.dirname(scene), extra)):
            if os.path.exists(cand):
                shutil.copy2(cand, dst)
                break
    m = os.path.join(scene, "masks")
    if os.path.isdir(m) and not os.path.islink(m) and not os.path.islink(os.path.join(m, sorted(os.listdir(m))[0], "0000.png")):
        shutil.copytree(m, os.path.join(dst, "masks"), dirs_exist_ok=True)
    n += 1
print("scenes:", n)
