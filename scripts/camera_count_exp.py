"""Camera-count experiment helpers.

prepare: split a 40-camera capture into held-out test cameras and nested,
         evenly spread training subsets; write one scene folder per subset
         (filtered COLMAP cameras + symlinks to the image folders).
evaluate: render a trained model from the held-out cameras and score it.

usage:
  python camera_count_exp.py prepare --src <scene> --out <exp dir> --n_test 5 --subsets 35,20,10,6,4
  python camera_count_exp.py evaluate --test_scene <exp>/test --rgb_source_path <dir> --ply <ply> --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

LIVESPLATS = "/scratch/hbommaka/code/basket-multiview/livesplats"
sys.path.insert(0, LIVESPLATS)


def look_at_point(centers, forwards):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in zip(centers, forwards):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def farthest_point_order(dirs, start=0):
    """Order unit vectors so every prefix is as evenly spread as possible."""
    order = [start]
    dist = 1 - dirs @ dirs[start]
    for _ in range(len(dirs) - 1):
        dist[order] = -1
        nxt = int(np.argmax(dist))
        order.append(nxt)
        dist = np.minimum(dist, 1 - dirs @ dirs[nxt])
    return order


def write_subset(src, dst, keep_ids, cams, imgs, write_model):
    os.makedirs(os.path.join(dst, "cameras"), exist_ok=True)
    sub_imgs = {k: v for k, v in imgs.items() if v.camera_id in keep_ids}
    sub_cams = {k: v for k, v in cams.items() if k in keep_ids}
    write_model(sub_cams, sub_imgs, {}, os.path.join(dst, "cameras"), ext=".bin")
    for name in os.listdir(src):
        if name != "cameras" and not os.path.exists(os.path.join(dst, name)):
            os.symlink(os.path.join(src, name), os.path.join(dst, name))


def prepare(args):
    from utils.colmap_utils import qvec2rotmat, read_model, write_model

    cams, imgs, _ = read_model(os.path.join(args.src, "cameras"), ext=".bin")
    ids, centers, forwards = [], [], []
    for im in imgs.values():
        R = qvec2rotmat(im.qvec)
        ids.append(im.camera_id)
        centers.append(-R.T @ im.tvec)
        forwards.append(R.T[:, 2])
    ids, centers, forwards = np.array(ids), np.array(centers), np.array(forwards)
    target = look_at_point(centers, forwards)
    dirs = centers - target
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    order = farthest_point_order(dirs)
    test = sorted(int(i) for i in ids[order[: args.n_test]])
    rest = [k for k in order if int(ids[k]) not in test]
    rest_order = [rest[j] for j in farthest_point_order(dirs[rest])]
    splits = {"test": test}
    for n in sorted((int(s) for s in args.subsets.split(",")), reverse=True):
        splits["n%02d" % n] = sorted(int(i) for i in ids[rest_order[:n]])

    for name, keep in splits.items():
        write_subset(args.src, os.path.join(args.out, name), set(keep), cams, imgs, write_model)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump({"look_at": target.tolist(), "camera_dirs": {int(i): d.tolist() for i, d in zip(ids, dirs)},
                   "splits": splits}, f, indent=1)
    for name, keep in splits.items():
        print("%-5s %2d cams: %s" % (name, len(keep), keep))


def evaluate(args):
    from types import SimpleNamespace

    import lpips
    import torch
    import torch.nn.functional as F
    from torchvision.utils import save_image

    import datasets.dataset_factory as dataset_factory
    from cs_renderers import cs_renderer_factory
    from losses._utils import ssim
    from scene_hs import GaussianModel

    conf = dict(dataset_factory.DATASET_CONF)
    conf.update(data_path=args.test_scene, timestamp=0, spatial_scale_factor=1, img_scale_factor=1.0,
                load_depth=False, require_all_img_data_available=False, dataset_caching=False,
                first_k_cams=-1, erosion_kernel_size=0, rgb_source_path=args.rgb_source_path)
    ds = dataset_factory.dataset_factory("basket_mv", conf)
    gm = GaussianModel(1)
    gm.load_ply(args.ply, 1.0)
    renderer = cs_renderer_factory(type="dynamic", conf={"background_mode": "black"})
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=True)
    lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    img_dir = os.path.splitext(args.out)[0] + "_views"
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            gt_rgb, alpha = s["im"].cuda(), s["alpha"].cuda()
            alpha = alpha[None] if alpha.dim() == 2 else alpha
            gt = gt_rgb * alpha  # subject on black, same as the render background
            sample = {"cam_info": {k: s["cam_info"][k] for k in ("width", "height", "K", "w2c")}}
            pred = renderer.render(sample, gm, pipe)["render"].clamp(0, 1)
            # score inside the subject's bounding box (padded) so empty background doesn't inflate PSNR
            ys, xs = torch.nonzero(alpha[0] > 0, as_tuple=True)
            pad = 20
            y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, gt.shape[1])
            x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, gt.shape[2])
            g, p = gt[:, y0:y1, x0:x1], pred[:, y0:y1, x0:x1]
            mse_box = F.mse_loss(p, g).item()
            mse_full = F.mse_loss(pred, gt).item()
            row = {"cam": s["id"],
                   "psnr_box": 10 * np.log10(1 / mse_box),
                   "psnr_full": 10 * np.log10(1 / mse_full),
                   "ssim_box": float(ssim(p[None], g[None])),
                   "lpips_box": float(lp(p[None] * 2 - 1, g[None] * 2 - 1))}
            rows.append(row)
            save_image(torch.cat([g, p], dim=2), os.path.join(img_dir, "cam%02d_gt_vs_pred.png" % s["id"]))
            save_image(p, os.path.join(img_dir, "cam%02d_pred.png" % s["id"]))
            save_image(g, os.path.join(img_dir, "cam%02d_gt.png" % s["id"]))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "cam"}
    out = {"ply": args.ply, "num_gaussians": int(gm.get_xyz.shape[0]), "mean": mean, "per_camera": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("mean:", {k: round(v, 4) for k, v in mean.items()}, "gaussians:", out["num_gaussians"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_test", type=int, default=5)
    p.add_argument("--subsets", default="35,20,10,6,4")
    e = sub.add_parser("evaluate")
    e.add_argument("--test_scene", required=True)
    e.add_argument("--rgb_source_path", required=True)
    e.add_argument("--ply", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else evaluate(a)
