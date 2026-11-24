"""Build scene folders for the overnight experiments.

cams:   a camera subset of a scene, optionally with calibration errors injected
        (rotation in degrees, camera-center shift in scene units, focal error in %).
sync:   a single-frame scene from a video sequence where some cameras show a
        later frame (simulated time-sync error).
layout: print a camera subset chosen by elevation / azimuth rules.

usage:
  python exp_variants.py cams   --src S --out O --ids 0,1,2 [--rot_deg X] [--trans X] [--focal_pct X] [--seed N]
  python exp_variants.py sync   --seq SEQ_DIR --rgb RGB_DIR --cams CAM_DIR --out O --ids ... --late_ids ... --t0 T --offset K
  python exp_variants.py layout --splits splits.json --rule upper|side|notop|ring0 [--n 10]
"""
import argparse
import json
import os
import sys

import numpy as np

LIVESPLATS = "/scratch/hbommaka/code/basket-multiview/livesplats"
sys.path.insert(0, LIVESPLATS)
from utils.colmap_utils import qvec2rotmat, read_model, rotmat2qvec, write_model  # noqa: E402


def axis_angle(axis, deg):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def link_all(src, dst, skip=("cameras",)):
    for name in os.listdir(src):
        if name not in skip and not os.path.exists(os.path.join(dst, name)):
            os.symlink(os.path.join(src, name), os.path.join(dst, name))


def cmd_cams(a):
    rng = np.random.default_rng(a.seed)
    cams, imgs, _ = read_model(os.path.join(a.src, "cameras"), ext=".bin")
    keep = set(int(i) for i in a.ids.split(","))
    out_imgs, out_cams, log = {}, {}, []
    for k, im in imgs.items():
        if im.camera_id not in keep:
            continue
        R = qvec2rotmat(im.qvec)
        C = -R.T @ im.tvec
        if a.rot_deg:
            R = axis_angle(rng.normal(size=3), a.rot_deg) @ R  # rotate camera about its own center
        if a.trans:
            d = rng.normal(size=3)
            C = C + a.trans * d / np.linalg.norm(d)
        out_imgs[k] = im._replace(qvec=rotmat2qvec(R), tvec=-R @ C)
        cam = cams[im.camera_id]
        params = np.array(cam.params, dtype=float)
        if a.focal_pct:
            sign = rng.choice([-1.0, 1.0])
            params[:2] *= 1 + sign * a.focal_pct / 100.0  # fx, fy (PINHOLE)
        out_cams[im.camera_id] = cam._replace(params=params)
        log.append(im.camera_id)
    os.makedirs(os.path.join(a.out, "cameras"), exist_ok=True)
    write_model(out_cams, out_imgs, {}, os.path.join(a.out, "cameras"), ext=".bin")
    link_all(a.src, a.out)
    json.dump(vars(a), open(os.path.join(a.out, "variant.json"), "w"), indent=1)
    print("wrote %s: %d cams, rot %.3f deg, trans %.2f, focal %.2f%%" % (a.out, len(log), a.rot_deg, a.trans, a.focal_pct))


def cmd_sync(a):
    ids = [int(i) for i in a.ids.split(",")]
    late = set(int(i) for i in a.late_ids.split(",")) if a.late_ids else set()
    cams, imgs, _ = read_model(a.cams, ext=".bin")
    keep = set(ids)
    os.makedirs(os.path.join(a.out, "cameras"), exist_ok=True)
    write_model({k: v for k, v in cams.items() if k in keep},
                {k: v for k, v in imgs.items() if v.camera_id in keep}, {}, os.path.join(a.out, "cameras"), ext=".bin")
    for cid in ids:
        t = a.t0 + (a.offset if cid in late else 0)
        for kind, src_root in (("rgb", a.rgb), ("normals", a.rgb), ("masks", os.path.join(a.seq, "masks"))):
            d = os.path.join(a.out, kind, "cam_%04d" % cid)
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, "0000.png")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.join(src_root, "cam_%04d" % cid, "%04d.png" % t), dst)
    json.dump(vars(a), open(os.path.join(a.out, "variant.json"), "w"), indent=1)
    print("wrote %s: %d cams, %d late by %d frames (t0=%d)" % (a.out, len(ids), len(late & keep), a.offset, a.t0))


def cmd_layout(a):
    s = json.load(open(a.splits))
    test = set(s["splits"]["test"])
    info = []
    for cid, d in s["camera_dirs"].items():
        d = np.array(d)
        info.append((int(cid), np.degrees(np.arcsin(d[2])), np.degrees(np.arctan2(d[1], d[0])), d))
    pool = [x for x in info if x[0] not in test]
    if a.rule == "upper":      # nothing below the subject (real venues: no cameras under the floor)
        pool = [x for x in pool if x[1] > -1]
    elif a.rule == "side":     # one side only, like a broadcast sideline (azimuth within +-91 deg, not below)
        pool = [x for x in pool if x[1] > -1 and abs(x[2]) < 91]
    elif a.rule == "notop":    # around the subject but no overhead views (elevation 0 and 30 rings)
        pool = [x for x in pool if -1 < x[1] < 45]
    elif a.rule == "ring0":    # a single ring at subject height
        pool = [x for x in pool if abs(x[1]) < 1]
    dirs = np.stack([x[3] for x in pool])
    order = [0]
    dist = 1 - dirs @ dirs[0]
    while len(order) < min(a.n, len(pool)):
        dist[order] = -1
        nxt = int(np.argmax(dist))
        order.append(nxt)
        dist = np.minimum(dist, 1 - dirs @ dirs[nxt])
    print(",".join(str(pool[i][0]) for i in sorted(order, key=lambda j: pool[j][0])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cams")
    c.add_argument("--src", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--ids", required=True)
    c.add_argument("--rot_deg", type=float, default=0.0)
    c.add_argument("--trans", type=float, default=0.0)
    c.add_argument("--focal_pct", type=float, default=0.0)
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sync")
    s.add_argument("--seq", required=True)
    s.add_argument("--rgb", required=True)
    s.add_argument("--cams", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ids", required=True)
    s.add_argument("--late_ids", default="")
    s.add_argument("--t0", type=int, required=True)
    s.add_argument("--offset", type=int, default=0)
    l = sub.add_parser("layout")
    l.add_argument("--splits", required=True)
    l.add_argument("--rule", required=True)
    l.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    {"cams": cmd_cams, "sync": cmd_sync, "layout": cmd_layout}[a.cmd](a)
