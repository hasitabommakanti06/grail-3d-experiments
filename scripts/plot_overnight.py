"""Charts + image strips for the overnight experiments.

usage: python plot_overnight.py <overnight dir> <out dir> [--cam 8]
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
S1, S2 = "#2a78d6", "#eb6834"
CC = "/scratch/hbommaka/output/exp_camera_count"
APOSE_MASKS = "/scratch/hbommaka/data/assets/players_A_pose/male/masks"
UPPER_TEST = {6, 8, 20}

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--cam", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def ev(path):
    p = os.path.join(path, "eval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def psnr(e, upper=False):
    if upper:
        v = [c["psnr_box"] for c in e["per_camera"] if c["cam"] in UPPER_TEST]
        return float(np.mean(v))
    return e["mean"]["psnr_box"]


baseline_runs = [e for e in (ev(os.path.join(CC, "n20")), ev(os.path.join(args.root, "baseline", "r2")),
                             ev(os.path.join(args.root, "baseline", "r3"))) if e]
base_vals = [psnr(e) for e in baseline_runs]
base_mean = float(np.mean(base_vals))
base_lo, base_hi = min(base_vals), max(base_vals)


def style(ax, title, xlabel, ylabel="PSNR on unseen cameras (dB)"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=14, pad=12)
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


def noise_band(ax):
    ax.axhspan(base_lo, base_hi, color=BAND, alpha=0.6, lw=0, zorder=1)
    ax.text(ax.get_xlim()[0], base_hi, "  run-to-run spread", va="bottom", ha="left", color=INK2, fontsize=9)


def line_chart(xs, ys, labels_at, title, xlabel, name, note, xticklabels=None, logx=False, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, ys, color=S1, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    for x, y in zip(xs, ys):
        if x in labels_at:
            ax.annotate("%.1f dB" % y, (x, y), textcoords="offset points", xytext=(0, 11), ha="center", color=INK, fontsize=11)
    if logx:
        ax.set_xscale("symlog", linthresh=min(v for v in xs if v > 0))
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels or [str(x) for x in xs])
    ax.minorticks_off()
    lo, hi = min(ys + [base_lo]), max(ys + [base_hi])
    ax.set_ylim(ylim or (lo - 1.5, hi + 2))
    style(ax, title, xlabel)
    noise_band(ax)
    save(fig, name, note)


def strip(tiles, name, h=420):
    tiles = [(lab, p) for lab, p in tiles if os.path.exists(p)]
    if not tiles:
        return
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


def view(path, kind="pred"):
    return os.path.join(path, "eval_views", "cam%02d_%s.png" % (args.cam, kind))


GT = view(os.path.join(CC, "n20"), "gt")
NOTE = "One standing player, 20 training cameras at ~2 m, 5 unseen test cameras, 10k iterations (BASKET-Multiview A-pose scan)."
summary = {"baseline_psnr_runs": base_vals}

# --- calibration -------------------------------------------------------------
for grp, unit, fmt, title, xlabel in (
        ("calib_rot", "deg", "%g°", "Camera aiming error", "Every camera rotated by (degrees)"),
        ("calib_trans", "cm", "%g cm", "Camera position error", "Every camera moved by (cm)"),
        ("calib_focal", "pct", "%g%%", "Zoom (focal length) error", "Every camera's focal length off by (%)")):
    pts = []
    for d in glob.glob(os.path.join(args.root, grp, unit + "*")):
        e = ev(d)
        if e:
            pts.append((float(os.path.basename(d)[len(unit):]), psnr(e), d))
    if not pts:
        continue
    pts.sort()
    xs, ys = [0.0] + [p[0] for p in pts], [base_mean] + [p[1] for p in pts]
    summary[grp] = dict(zip(map(str, xs), ys))
    line_chart(xs, ys, {xs[0], xs[-1]} | {p[0] for p in pts[len(pts) // 2:len(pts) // 2 + 1]}, title, xlabel,
               grp + ".png", NOTE, [fmt % x for x in xs], logx=True)
    strip([("Real camera", GT), ("Perfect calibration", view(os.path.join(CC, "n20")))] +
          [(fmt % x, view(d)) for x, _, d in pts], grp + "_strip.png")

# --- all calibration errors on one axis: mean reprojection error in pixels -------
def load_cams(scene):
    import sys as _s
    _s.path.insert(0, "/scratch/hbommaka/code/basket-multiview/livesplats")
    from utils.colmap_utils import qvec2rotmat, read_model
    cams, imgs, _ = read_model(os.path.join(scene, "cameras"), ext=".bin")
    out = {}
    for im in imgs.values():
        c = cams[im.camera_id]
        K = np.array([[c.params[0], 0, c.params[2]], [0, c.params[1], c.params[3]], [0, 0, 1]])
        out[im.camera_id] = (K, qvec2rotmat(im.qvec), np.array(im.tvec))
    return out


def project(cam, X):
    K, R, t = cam
    pc = X @ R.T + t
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


hull = np.array([[float(v) for v in ln.split()[1:4]] for ln in open(os.path.join(CC, "n20", "hull.obj")) if ln.startswith("v ")])
hull[:, :2] *= -1  # carve.py pre-negates x, y for load_obj
clean = load_cams(os.path.join(CC, "n20"))
series = {}
for grp, unit, lab in (("calib_rot", "deg", "Aiming (rotation)"), ("calib_trans", "cm", "Position"), ("calib_focal", "pct", "Zoom (focal length)")):
    pts = []
    for d in glob.glob(os.path.join(args.root, grp, unit + "*")):
        e = ev(d)
        if not e:
            continue
        noisy = load_cams(os.path.join(d, "scene"))
        err = np.mean([np.linalg.norm(project(noisy[k], hull) - project(clean[k], hull), axis=1).mean() for k in noisy])
        pts.append((float(err), psnr(e), os.path.basename(d)[len(unit):]))
    if pts:
        series[lab] = sorted(pts)
if series:
    summary["calibration_px"] = series
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for (lab, pts), col in zip(series.items(), (S1, S2, "#1baf7a")):
        xs, ys = [0.3] + [p[0] for p in pts], [base_mean] + [p[1] for p in pts]
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=lab)
        off, ha = ((0, -20), "center") if lab.startswith("Zoom") else ((8, -4), "left")
        ax.annotate(lab, (xs[-1], ys[-1]), textcoords="offset points", xytext=off, ha=ha, color=INK2, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0 (perfect)", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(0.25, 60)
    style(ax, "Calibration error, measured in pixels", "Average reprojection error (pixels, log scale)")
    ax.axvspan(0.25, 1, color=BAND, alpha=0.5, lw=0, zorder=0)
    ax.text(0.5, ax.get_ylim()[1], "under 1 px", ha="center", va="top", color=INK2, fontsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK2)
    save(fig, "calibration_pixels.png", NOTE + " Pixel error = how far a body point moves in the image because of the calibration error.")

# --- resolution: x = player height in pixels ---------------------------------
heights = []
for f in sorted(glob.glob(os.path.join(APOSE_MASKS, "cam_*", "0000.png")))[:40]:
    m = np.asarray(Image.open(f).convert("L")) > 0
    rows = np.nonzero(m.any(axis=1))[0]
    if len(rows):
        heights.append(rows[-1] - rows[0] + 1)
full_h = float(np.median(heights))
pts = [(1.0, base_mean, os.path.join(CC, "n20"))]
for d in glob.glob(os.path.join(args.root, "resolution", "scale*")):
    e = ev(d)
    if e:
        pts.append((float(os.path.basename(d)[5:]), psnr(e), d))
pts.sort()
if len(pts) > 1:
    hs = [round(full_h * s) for s, _, _ in pts]
    ys = [p[1] for p in pts]
    summary["resolution"] = {"player_px_full": full_h, "points": [(s, h, y) for (s, _, _), h, y in zip(pts, hs, ys)]}
    line_chart(hs, ys, {hs[0], hs[-1]}, "Pixels on the player", "Player height in the training images (pixels)",
               "resolution.png", NOTE + " Rendered and scored at full resolution.", [str(h) for h in hs], logx=True)
    strip([("Real camera", GT)] + [("%d px tall" % h, view(d)) for (s, _, d), h in sorted(zip(pts, hs), key=lambda t: -t[1])],
          "resolution_strip.png")

# --- layout: grouped bars, all test cams vs. realistic (above-floor) test cams --
names = [("surround", os.path.join(CC, "n10"), "All around\n(incl. below)"),
         ("upper", os.path.join(args.root, "layout", "upper"), "Upper half\nonly"),
         ("notop", os.path.join(args.root, "layout", "notop"), "No overhead\n(0° & 30°)"),
         ("side", os.path.join(args.root, "layout", "side"), "One side\nonly"),
         ("ring0", os.path.join(args.root, "layout", "ring0"), "One ring at\nplayer height (7)")]
rows = [(lab, ev(p), p) for _, p, lab in names if ev(p)]
back = {p: json.load(open(os.path.join(p, "eval_back.json"))) for _, p, _ in names if os.path.exists(os.path.join(p, "eval_back.json"))}
if len(rows) > 1:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(rows))
    w = 0.38
    if len(back) == len(rows):
        a = [psnr(e, upper=True) for _, e, _ in rows]
        b = [back[p]["mean"]["psnr_box"] for _, _, p in rows]
        la, lb = "Unseen views from the front half", "Unseen views from behind"
    else:
        a = [psnr(e) for _, e, _ in rows]
        b = [psnr(e, upper=True) for _, e, _ in rows]
        la, lb = "All 5 unseen cameras", "3 unseen cameras at or above the player"
    ax.bar(x - w / 2 - 0.01, a, w, color=S1, label=la, zorder=3)
    ax.bar(x + w / 2 + 0.01, b, w, color=S2, label=lb, zorder=3)
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax.text(xi - w / 2, va + 0.15, "%.1f" % va, ha="center", color=INK, fontsize=10)
        ax.text(xi + w / 2, vb + 0.15, "%.1f" % vb, ha="center", color=INK, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, max(a + b) + 2)
    style(ax, "Where the cameras are (10 cameras each)", "")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10, labelcolor=INK2)
    ax.set_title("Where the cameras are (10 cameras each)", loc="left", color=INK, fontsize=14, pad=34)
    summary["layout"] = {r[0].replace("\n", " "): {la: va, lb: vb} for r, va, vb in zip(rows, a, b)}
    save(fig, "layout.png", "Same player and 10k iterations. 'One ring' has only 7 cameras and its model collapsed (6 Gaussians left after cleanup). "
         "Front = held-out cameras 6, 8, 20 (0-60 deg up, facing the player's front/sides); behind = cameras 9, 10, 18 "
         "(30-60 deg up, behind the player), unused by every layout.")
    strip([("Real camera", GT)] + [(lab.replace("\n", " "), view(p)) for lab, _, p in rows], "layout_strip.png")

# --- sync ---------------------------------------------------------------------
pts = []
for d in glob.glob(os.path.join(args.root, "sync", "lag*")):
    e = ev(d)
    if e:
        pts.append((int(os.path.basename(d)[3:]), psnr(e), d))
pts.sort()
if len(pts) > 1:
    ms = [round(k * 1000 / 30) for k, _, _ in pts]
    ys = [p[1] for p in pts]
    summary["sync"] = dict(zip(map(str, ms), ys))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ms, ys, color=S1, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    for xv, yv in zip(ms, ys):
        ax.annotate("%.1f dB" % yv, (xv, yv), textcoords="offset points", xytext=(0, 11), ha="center", color=INK, fontsize=11)
    ax.set_xticks(ms)
    ax.set_xticklabels(["in sync" if v == 0 else "%d ms\n(%d frame%s)" % (v, k, "s" if k > 1 else "") for v, (k, _, _) in zip(ms, pts)])
    ax.set_ylim(min(ys) - 1.5, max(ys) + 2)
    style(ax, "Time-sync error (half the cameras lag)", "How late half of the 20 training cameras fire")
    save(fig, "sync.png", "Player jogging in place (BASKET-Multiview 'running_black'); feet move ~20 cm per 33 ms frame.")
    strip([("Real camera", view(pts[0][2], "gt"))] + [("in sync" if k == 0 else "%d ms late" % m, view(d)) for (k, _, d), m in zip(pts, ms)],
          "sync_strip.png")

# --- iterations -----------------------------------------------------------------
pts = [(10000, base_mean, os.path.join(CC, "n20"))]
for d in glob.glob(os.path.join(args.root, "iterations", "it*")):
    e = ev(d)
    if e:
        pts.append((int(os.path.basename(d)[2:]), psnr(e), d))
pts.sort()
if len(pts) > 1:
    secs = []
    for it, _, d in pts:
        rj = os.path.join(d, "run.json")
        secs.append(json.load(open(rj))["train_seconds"] if os.path.exists(rj) else None)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    summary["iterations"] = [(i, y, s) for i, y, s in zip(xs, ys, secs)]
    labels = ["%s\n(%s)" % ("{:,}".format(i), ("%d s" % s) if s and s < 120 else ("%.0f min" % (s / 60)) if s else "~6 min")
              for i, s in zip(xs, secs)]
    line_chart(xs, ys, set(xs), "Training steps (compute per frame)", "Training steps (wall time on one Tesla T4)",
               "iterations.png", NOTE.replace("10k iterations", "varying iterations"), labels, logx=True)
    strip([("Real camera", GT)] + [("{:,} steps".format(i), view(d)) for i, _, d in pts], "iterations_strip.png")

json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print("figures:", sorted(os.listdir(args.out)))
