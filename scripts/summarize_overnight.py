"""Collect every overnight eval.json into one CSV + a readable table.

usage: python summarize_overnight.py <overnight dir>
"""
import json
import os
import sys

root = sys.argv[1]
UPPER_TEST = {6, 8, 20}  # held-out cameras at or above the subject (realistic viewpoints)
rows = []
# reference points from the camera-count run (same 20 cameras, 10k iterations)
cc = "/scratch/hbommaka/output/exp_camera_count"
for n in ("n20", "n10"):
    p = os.path.join(cc, n, "eval.json")
    if os.path.exists(p):
        rows.append(("reference", n, json.load(open(p)), {}))
for g in sorted(os.listdir(root)):
    gd = os.path.join(root, g)
    if not os.path.isdir(gd):
        continue
    for n in sorted(os.listdir(gd)):
        p = os.path.join(gd, n, "eval.json")
        if os.path.exists(p):
            rj = os.path.join(gd, n, "run.json")
            rows.append((g, n, json.load(open(p)), json.load(open(rj)) if os.path.exists(rj) else {}))

hdr = "group,name,psnr_box,psnr_box_upper_cams,ssim_box,lpips_box,num_gaussians,train_seconds"
lines = [hdr]
for g, n, e, r in rows:
    up = [c["psnr_box"] for c in e["per_camera"] if c["cam"] in UPPER_TEST]
    lines.append("%s,%s,%.2f,%.2f,%.4f,%.4f,%d,%s" % (
        g, n, e["mean"]["psnr_box"], sum(up) / len(up) if up else float("nan"),
        e["mean"]["ssim_box"], e["mean"]["lpips_box"], e["num_gaussians"], r.get("train_seconds", "")))
open(os.path.join(root, "summary.csv"), "w").write("\n".join(lines) + "\n")
for ln in lines:
    print("  ".join("%-14s" % x for x in ln.split(",")))
