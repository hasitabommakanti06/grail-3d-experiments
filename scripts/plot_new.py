"""Charts + image strips for the Nov 24 experiments (occlusion, mask quality, sync follow-up).

usage: python plot_new.py <output dir containing new/, overnight/, exp_camera_count/> <out dir> [--cam 8]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--cam", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
NEW, OLD, CC = (os.path.join(args.root, d) for d in ("new", "overnight", "exp_camera_count"))


def P(*parts):
    return json.load(open(os.path.join(*parts, "eval.json")))["mean"]["psnr_box"]


def view(path, kind="pred"):
    return os.path.join(path, "eval_views", "cam%02d_%s.png" % (args.cam, kind))


def style(ax, title, xlabel, ylabel="PSNR on unseen cameras (dB)"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)


def save(fig, name, note):
    import textwrap
    fig.patch.set_facecolor(SURFACE)
    wrapped = textwrap.fill(note, width=int(fig.get_figwidth() * 15))
    lines = wrapped.count("\n") + 1
    fig.text(0.01, 0.01, wrapped, color=INK2, fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.035 * lines + 0.01, 1, 1))
    fig.savefig(os.path.join(args.out, name), facecolor=SURFACE, dpi=200)
    plt.close(fig)


def line(ax, xs, ys, color=S1, label=None, fmt="%.1f", above=True, skip=0, dx=0):
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=label)
    for x, y in list(zip(xs, ys))[skip:]:
        ax.annotate(fmt % y, (x, y), textcoords="offset points", xytext=(dx, 10 if above else -17), ha="center", color=INK, fontsize=10)


def strip(tiles, name, h=420):
    tiles = [(lab, p) for lab, p in tiles if os.path.exists(p)]
    imgs = [Image.open(p).convert("RGB") for _, p in tiles]
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    label_h, gap = 40, 6
    W = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (W, h + label_h), SURFACE)
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for (lab, _), im in zip(tiles, imgs):
        out.paste(im, (x, label_h))
        d.text((x + (im.width - d.textlength(lab, font=font)) / 2, 9), lab, fill=INK, font=font)
        x += im.width + gap
    out.save(os.path.join(args.out, name))


base = float(np.mean([P(CC, "n20"), P(OLD, "baseline", "r2"), P(OLD, "baseline", "r3")]))
GT = view(os.path.join(CC, "n20"), "gt")
NOTE = "One standing player, 20 training cameras, 5 unseen test cameras, 10k steps. Test images always use the true masks."
summary = {"baseline": base}

# --- occlusion ------------------------------------------------------------------
occ = os.path.join(NEW, "occlusion")
ctrl = P(occ, "cams0_noprune")
cams = [(0, ctrl)] + [(f, P(occ, "cams%d_cover50" % f)) for f in (25, 50, 75, 100)]
cover = [(0, ctrl)] + [(c, P(occ, "cams50_cover%d" % c)) for c in (25, 50, 75)]
summary["occlusion"] = {"cameras_blocked_pct (half the player hidden)": cams, "player_hidden_pct (in half the cameras)": cover}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
line(axs[0], [c[0] for c in cams], [c[1] for c in cams])
axs[0].set_xticks([c[0] for c in cams])
axs[0].set_xticklabels(["none", "25%", "50%", "75%", "all"])
style(axs[0], "Half the player hidden in some cameras", "Cameras where the player is blocked")
line(axs[1], [c[0] for c in cover], [c[1] for c in cover], color=S2)
axs[1].set_xticks([c[0] for c in cover])
axs[1].set_xticklabels(["none", "25%", "50%", "75%"])
style(axs[1], "Blocked in half the cameras", "How much of the player is hidden", ylabel="")
axs[0].set_ylim(min(c[1] for c in cams) - 2, base + 2.5)
save(fig, "occlusion.png", NOTE + " Occlusion is simulated by cutting the hidden part out of the mask in the blocked cameras. "
     "The released code crashes on this, so these runs skip its final prune-by-masks step.")
strip([("Real camera", GT), ("Not blocked", view(os.path.join(occ, "cams0_noprune")))] +
      [("%d%% of cameras" % f if f < 100 else "All cameras", view(os.path.join(occ, "cams%d_cover50" % f))) for f in (25, 50, 75, 100)],
      "occlusion_strip.png")

# --- mask quality -----------------------------------------------------------------
def shift(*parts):
    return json.load(open(os.path.join(NEW, *parts, "scene", "variant.json")))["mean_edge_shift_px"]

er = [(0, base)] + [(x, P(NEW, "mask_erode", "px%d" % x)) for x in (2, 5, 10)]
di = [(0, base)] + [(x, P(NEW, "mask_dilate", "px%d" % x)) for x in (2, 5, 10)]
ji = [(0, base)] + [(shift("mask_jitter", "px%d" % x), P(NEW, "mask_jitter", "px%d" % x)) for x in (4, 8)]
holes = P(NEW, "mask_holes", "r40")
summary["masks"] = {"too_tight_px": er, "too_loose_px": di, "wobbly_edge_px": ji, "three_40px_holes": holes}
fig, ax = plt.subplots(figsize=(8.4, 4.8))
line(ax, [p[0] for p in di], [p[1] for p in di], color=S3, label="Mask too loose (grown)")
line(ax, [p[0] for p in ji], [p[1] for p in ji], color=S2, label="Wobbly edge (in and out)", skip=1, dx=14)
line(ax, [p[0] for p in er], [p[1] for p in er], color=S1, label="Mask too tight (shrunk)", above=False, skip=1)
ax.set_xticks([0, 2, 5, 10])
ax.set_xticklabels(["perfect", "2 px", "5 px", "10 px"])
ax.set_ylim(min(p[1] for p in er) - 2.5, base + 2)
style(ax, "Mask errors", "How far the mask edge is off (pixels, player is ~808 px tall)")
ax.legend(frameon=False, loc="lower left", fontsize=10, labelcolor=INK2)
save(fig, "masks.png", NOTE + " Three 40 px holes in every mask gave %.1f dB. The image background here is black, so a loose mask only lets in "
     "black pixels. Real backgrounds would likely hurt more." % holes)
strip([("Real camera", GT), ("Perfect masks", view(os.path.join(CC, "n20"))),
       ("5 px too loose", view(os.path.join(NEW, "mask_dilate", "px5"))),
       ("Wobbly edge", view(os.path.join(NEW, "mask_jitter", "px8"))),
       ("5 px too tight", view(os.path.join(NEW, "mask_erode", "px5"))),
       ("10 px too tight", view(os.path.join(NEW, "mask_erode", "px10"))),
       ("Holes", view(os.path.join(NEW, "mask_holes", "r40")))], "masks_strip.png")

# --- sync follow-up ------------------------------------------------------------------
sm = os.path.join(NEW, "sync_motion")
mot = json.load(open(os.path.join(sm, "motion.json")))["edge_shift_px_next_frame"]
motd = json.load(open(os.path.join(sm, "motion_dribbling.json")))
pts = []
for t in json.load(open(os.path.join(sm, "motion.json")))["picked_t0"]:
    pts.append((mot[str(t)], P(sm, "t%d_lag0" % t), P(sm, "t%d_lag1" % t), "jog"))
pts.append((mot["20"], P(OLD, "sync", "lag0"), P(OLD, "sync", "lag1"), "jog"))
td = motd["picked_t0"][0]
pts.append((motd["edge_shift_px_next_frame"][str(td)], P(sm, "drib_t%d_lag0" % td), P(sm, "drib_t%d_lag1" % td), "dribble"))
pts.sort()
count = [(0, P(OLD, "sync", "lag0")), (1, P(NEW, "sync_count", "late1")), (5, P(NEW, "sync_count", "late5")), (10, P(OLD, "sync", "lag1"))]
summary["sync_motion"] = [{"edge_px": p[0], "in_sync": p[1], "one_frame_late": p[2], "clip": p[3]} for p in pts]
summary["sync_count"] = count
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
xs = [p[0] for p in pts]
axs[0].plot(xs, [p[1] for p in pts], color=INK2, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label="All cameras in sync")
line(axs[0], xs, [p[2] for p in pts], label="Half the cameras 1 frame late", above=False)
axs[0].set_xlim(4.5, 17)
style(axs[0], "One frame late, at slower and faster moments", "How far the player's outline moves in one frame (pixels)")
axs[0].legend(frameon=False, loc="center right", fontsize=10, labelcolor=INK2)
line(axs[1], [0, 1, 2, 3], [c[1] for c in count], color=S2)
axs[1].set_xticks([0, 1, 2, 3])
axs[1].set_xticklabels(["none", "1 of 20", "5 of 20", "10 of 20"])
style(axs[1], "How many cameras are late", "Cameras that are 1 frame (33 ms) late", ylabel="")
axs[0].set_ylim(min(p[2] for p in pts) - 2.5, max(p[1] for p in pts) + 2)
save(fig, "sync_followup.png", "Jogging player (one point from the dribbling clip, the slowest motion in the dataset), 20 training cameras, "
     "5 unseen test cameras, 10k steps. Frames are 33 ms apart, so offsets smaller than one frame could not be tested directly.")
strip([("Real camera", view(os.path.join(OLD, "sync", "lag0"), "gt")), ("In sync", view(os.path.join(OLD, "sync", "lag0"))),
       ("1 camera late", view(os.path.join(NEW, "sync_count", "late1"))),
       ("5 cameras late", view(os.path.join(NEW, "sync_count", "late5"))),
       ("10 cameras late", view(os.path.join(OLD, "sync", "lag1")))], "sync_followup_strip.png")

json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
