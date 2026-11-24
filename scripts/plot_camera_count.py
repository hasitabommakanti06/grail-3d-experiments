"""Chart + image strip for the camera-count experiment.

usage: python plot_camera_count.py <exp dir> <out dir> [--cam 8]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("out")
ap.add_argument("--cam", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

runs = []
for name in sorted(os.listdir(args.exp)):
    p = os.path.join(args.exp, name, "eval.json")
    if name.startswith("n") and os.path.exists(p):
        runs.append((int(name[1:]), json.load(open(p))))
runs.sort()
ns = [n for n, _ in runs]

with open(os.path.join(args.out, "camera_count_results.csv"), "w") as f:
    f.write("train_cameras,psnr_box_db,psnr_full_db,ssim_box,lpips_box,num_gaussians\n")
    for n, r in runs:
        m = r["mean"]
        f.write("%d,%.2f,%.2f,%.4f,%.4f,%d\n" % (n, m["psnr_box"], m["psnr_full"], m["ssim_box"], m["lpips_box"], r["num_gaussians"]))

# --- chart: held-out PSNR vs number of training cameras -------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ys = [r["mean"]["psnr_box"] for _, r in runs]
ax.plot(ns, ys, color=SERIES, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
for n, y in zip(ns, ys):
    if n in (ns[0], 10, ns[-1]):
        ax.annotate("%.1f dB" % y, (n, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", color=INK, fontsize=11)
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()
ax.set_xlabel("Training cameras (evenly spread around the player)", color=INK2)
ax.set_ylabel("PSNR on 5 unseen cameras (dB)", color=INK2)
ax.set_ylim(min(ys) - 2, max(ys) + 2.5)
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.set_title("Quality on unseen viewpoints vs. number of cameras", loc="left", color=INK, fontsize=14, pad=14)
fig.text(0.01, 0.01, "One standing player, 40-camera dome at ~2 m (BASKET-Multiview A-pose scan); 10k iterations per run.",
         color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(args.out, "camera_count_psnr.png"), facecolor=SURFACE)

# --- strip: the same unseen camera, real vs. each model --------------------
tiles = [("Real camera", os.path.join(args.exp, "n%02d" % ns[-1], "eval_views", "cam%02d_gt.png" % args.cam))]
for n, r in reversed(runs):
    tiles.append(("%d cams · %.1f dB" % (n, r["mean"]["psnr_box"]),
                  os.path.join(args.exp, "n%02d" % n, "eval_views", "cam%02d_pred.png" % args.cam)))
imgs = [Image.open(p).convert("RGB") for _, p in tiles]
h = 520
imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
label_h, gap = 44, 6
W = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
strip = Image.new("RGB", (W, h + label_h), SURFACE)
draw = ImageDraw.Draw(strip)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
except OSError:
    font = ImageFont.load_default()
x = 0
for (label, _), im in zip(tiles, imgs):
    strip.paste(im, (x, label_h))
    tw = draw.textlength(label, font=font)
    draw.text((x + (im.width - tw) / 2, 10), label, fill=INK, font=font)
    x += im.width + gap
strip.save(os.path.join(args.out, "camera_count_strip_cam%02d.png" % args.cam))
print("wrote", os.listdir(args.out))
