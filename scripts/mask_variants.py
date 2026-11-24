"""Build scene folders with degraded player masks (the dataset's masks are perfect).

scene:  copy of a camera-subset scene whose masks are changed in every training camera
        erode / dilate   shrink or grow the mask by --px pixels
        jitter           wobbly boundary, moves in and out by about --px pixels
        holes            --n_holes round holes of radius --px punched into the mask
        occlude          in --frac of the cameras, hide --cover of the player's pixels
                         behind a vertical edge (as if another player stood in front)
motion: for a video sequence, how far the silhouette edge moves between frame t and t+1
        (pixels, averaged over cameras). Used to pick frames for the sync test.

usage:
  python mask_variants.py scene --src S --mask_src M --out O --mode erode --px 5 [--seed N]
  python mask_variants.py scene --src S --mask_src M --out O --mode occlude --frac 0.5 --cover 0.5
  python mask_variants.py motion --masks SEQ/masks --ids 1,2,3 --out motion.json
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

LIVESPLATS = "/scratch/hbommaka/code/basket-multiview/livesplats"
sys.path.insert(0, LIVESPLATS)


def load(path):
    return cv2.imread(path, cv2.IMREAD_COLOR).any(2)


def perimeter(m):
    k = np.ones((3, 3), np.uint8)
    return int((m & ~cv2.erode(m.astype(np.uint8), k).astype(bool)).sum())


def edge_shift(a, b):
    """Approximate average distance (px) the silhouette edge moved between masks a and b."""
    return float((a ^ b).sum()) / max(perimeter(a), 1)


def degrade(m, a, rng, occluded):
    u8 = m.astype(np.uint8)
    if a.mode in ("erode", "dilate"):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.px + 1, 2 * a.px + 1))
        return (cv2.erode(u8, k) if a.mode == "erode" else cv2.dilate(u8, k)).astype(bool)
    if a.mode == "jitter":
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), a.px)
        noise = cv2.GaussianBlur(rng.standard_normal(m.shape).astype(np.float32), (0, 0), 4 * a.px)
        out = (soft + 0.35 * noise / noise.std()) > 0.5
        near_edge = (soft > 0.02) & (soft < 0.98)   # only the boundary moves, no stray blobs
        return np.where(near_edge, out, m)
    if a.mode == "holes":
        out = u8.copy()
        ys, xs = np.nonzero(m)
        for j in rng.choice(len(ys), a.n_holes, replace=False):
            cv2.circle(out, (int(xs[j]), int(ys[j])), a.px, 0, -1)
        return out.astype(bool)
    if a.mode == "occlude":
        if not occluded:
            return m
        cols = np.cumsum(m.sum(0)) / m.sum()
        out = m.copy()
        if rng.random() < 0.5:   # hide the left part
            out[:, : int(np.searchsorted(cols, a.cover))] = False
        else:                    # hide the right part
            out[:, int(np.searchsorted(cols, 1 - a.cover)):] = False
        return out
    raise ValueError(a.mode)


def cmd_scene(a):
    from utils.colmap_utils import read_model

    rng = np.random.default_rng(a.seed)
    _, imgs, _ = read_model(os.path.join(a.src, "cameras"), ext=".bin")
    ids = sorted(im.camera_id for im in imgs.values())
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "cameras")):
        shutil.copytree(os.path.join(a.src, "cameras"), os.path.join(a.out, "cameras"))
    for name in os.listdir(a.src):
        if name in ("rgb", "normals", "depth", "true_depth") and not os.path.lexists(os.path.join(a.out, name)):
            os.symlink(os.path.realpath(os.path.join(a.src, name)), os.path.join(a.out, name))
    hidden = set(int(i) for i in rng.choice(ids, int(round(a.frac * len(ids))), replace=False)) if a.mode == "occlude" else set()
    stats = {}
    for cid in ids:
        m = load(os.path.join(a.mask_src, "cam_%04d" % cid, "%04d.png" % a.frame))
        out = degrade(m, a, rng, cid in hidden)
        d = os.path.join(a.out, "masks", "cam_%04d" % cid)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "0000.png"), np.repeat(out[:, :, None], 3, 2).astype(np.uint8) * 255)
        stats[cid] = {"pixels_before": int(m.sum()), "pixels_after": int(out.sum()),
                      "pixels_changed": int((m ^ out).sum()), "edge_shift_px": round(edge_shift(m, out), 2)}
    info = dict(vars(a), occluded_cameras=sorted(hidden), per_camera=stats,
                mean_edge_shift_px=float(np.mean([s["edge_shift_px"] for s in stats.values()])),
                mean_frac_pixels_changed=float(np.mean([s["pixels_changed"] / s["pixels_before"] for s in stats.values()])))
    json.dump(info, open(os.path.join(a.out, "variant.json"), "w"), indent=1)
    print("wrote %s: %s, %d cams, %.1f%% of mask pixels changed, edge moved %.1f px"
          % (a.out, a.mode, len(ids), 100 * info["mean_frac_pixels_changed"], info["mean_edge_shift_px"]))


def cmd_motion(a):
    ids = [int(i) for i in a.ids.split(",")]
    n = len(os.listdir(os.path.join(a.masks, "cam_%04d" % ids[0])))
    prev = {c: load(os.path.join(a.masks, "cam_%04d" % c, "0000.png")) for c in ids}
    shift = {}
    for t in range(1, n):
        cur = {c: load(os.path.join(a.masks, "cam_%04d" % c, "%04d.png" % t)) for c in ids}
        shift[t - 1] = float(np.mean([edge_shift(prev[c], cur[c]) for c in ids]))
        prev = cur
    # frames spread from the slowest to the fastest moment (keep clear of the sequence ends)
    ok = sorted((s, t) for t, s in shift.items() if 2 <= t <= n - 6)
    picks = [ok[int(round(q * (len(ok) - 1)))] for q in (0.0, 0.3, 0.65, 1.0)]
    json.dump({"edge_shift_px_next_frame": shift, "picked_t0": [t for _, t in picks]}, open(a.out, "w"), indent=1)
    print(",".join(str(t) for _, t in picks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scene")
    s.add_argument("--src", required=True)
    s.add_argument("--mask_src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", required=True, choices=["erode", "dilate", "jitter", "holes", "occlude"])
    s.add_argument("--px", type=int, default=0)
    s.add_argument("--n_holes", type=int, default=3)
    s.add_argument("--frac", type=float, default=0.0)
    s.add_argument("--cover", type=float, default=0.5)
    s.add_argument("--frame", type=int, default=0)
    s.add_argument("--seed", type=int, default=1)
    m = sub.add_parser("motion")
    m.add_argument("--masks", required=True)
    m.add_argument("--ids", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_scene(a) if a.cmd == "scene" else cmd_motion(a)
