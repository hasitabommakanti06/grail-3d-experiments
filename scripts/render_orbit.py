"""Render a 360-degree orbit around a trained Gaussian model (novel views).

The orbit circles the scene at the rig cameras' average distance and height,
so most frames are viewpoints where no real camera stood.

usage: python render_orbit.py --source_path <scene with cameras/> --ply <model.ply>
                              --out <dir> [--frames 120] [--scale 0.5]
"""
import argparse
import os
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, "/scratch/hbommaka/code/basket-multiview/livesplats")
import datasets.dataset_factory as dataset_factory  # noqa: E402
from cs_renderers import cs_renderer_factory  # noqa: E402
from scene_hs import GaussianModel  # noqa: E402


def look_at_w2c(eye, target, up):
    """OpenCV convention (x right, y down, z forward), returns 3x4 world-to-camera."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows = camera axes in world coords
    return np.hstack([R, (-R @ eye)[:, None]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--scale", type=float, default=0.5, help="image scale vs. dataset resolution")
    ap.add_argument("--sh_degree", type=int, default=1)
    ap.add_argument("--radius_scale", type=float, default=1.0, help=">1 pulls the orbit back from the subject")
    ap.add_argument("--rgb_source_path", default="", help="needed when rgb/ has clothing/skin subfolders")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conf = dict(dataset_factory.DATASET_CONF)
    ts = sorted(os.listdir(os.path.join(args.source_path, "masks", "cam_0000")))[0][:4]
    conf.update(data_path=args.source_path, timestamp=int(ts), spatial_scale_factor=1,
                img_scale_factor=args.scale, load_depth=False, require_all_img_data_available=False,
            dataset_caching=False, first_k_cams=-1, erosion_kernel_size=0,
                rgb_source_path=args.rgb_source_path)
    ds = dataset_factory.dataset_factory("basket_mv", conf)
    cams = ds.data

    centers = np.stack([c["extrinsics"]["c2w"][:, 3] for c in cams])
    up = -np.mean([c["extrinsics"]["c2w"][:, 1] for c in cams], axis=0)
    up /= np.linalg.norm(up)

    gm = GaussianModel(args.sh_degree)
    gm.load_ply(args.ply, 1.0)
    target = gm.get_xyz.detach().median(dim=0).values.cpu().numpy()

    rel = centers - target
    height = float(np.mean(rel @ up))
    radius = args.radius_scale * float(np.mean(np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)))
    e1 = rel[0] - (rel[0] @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    K = cams[0]["intrinsics"]
    sample0 = ds[0]
    W, H = sample0["cam_info"]["width"], sample0["cam_info"]["height"]
    renderer = cs_renderer_factory(type="dynamic", conf={"background_mode": "black"})
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=True)

    frames = []
    with torch.no_grad():
        for i in range(args.frames):
            th = 2 * np.pi * i / args.frames
            eye = target + radius * (np.cos(th) * e1 + np.sin(th) * e2) + height * up
            sample = {"cam_info": {"width": W, "height": H, "K": K,
                                   "w2c": look_at_w2c(eye, target, up).astype(np.float32)}}
            img = renderer.render(sample, gm, pipe)["render"].clamp(0, 1)
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(args.out, "orbit_%03d.png" % i), img)
            frames.append(img)

    imageio.mimsave(os.path.join(args.out, "orbit.mp4"), frames, fps=30, quality=8)
    small = [f[::3, ::3] for f in frames[::2]]
    imageio.mimsave(os.path.join(args.out, "orbit.gif"), small, duration=1 / 15, loop=0)
    print("wrote %d frames (%dx%d), radius %.2f, height %.2f -> %s" % (len(frames), W, H, radius, height, args.out))


if __name__ == "__main__":
    main()
