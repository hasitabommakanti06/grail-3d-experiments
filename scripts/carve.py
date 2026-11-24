"""Visual-hull (space carving) initialization from multi-view masks.

Replaces the missing objs/0000.obj: keeps every voxel that projects inside the
subject's mask in (almost) every camera, then writes the hull's surface voxels
as OBJ vertices that train_dynamic_bind.py --first_frame_obj can load.

usage: python carve.py <scene dir> <out.obj> [--rgb_source_path DIR] [--voxel 1.5] [--min_views_frac 0.95]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/scratch/hbommaka/code/basket-multiview/livesplats")
import datasets.dataset_factory as dataset_factory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("out")
ap.add_argument("--rgb_source_path", default="", help="needed when rgb/ has clothing/skin subfolders")
ap.add_argument("--voxel", type=float, default=1.5, help="voxel size in scene units (cm here)")
ap.add_argument("--min_views_frac", type=float, default=0.95)
ap.add_argument("--img_scale", type=float, default=1.0, help="carve from downscaled masks")
args = ap.parse_args()

conf = dict(dataset_factory.DATASET_CONF)
conf.update(data_path=args.scene, timestamp=0, spatial_scale_factor=1, img_scale_factor=args.img_scale,
            load_depth=False, require_all_img_data_available=False,
            dataset_caching=False, first_k_cams=-1, erosion_kernel_size=0,
            rgb_source_path=args.rgb_source_path)
ds = dataset_factory.dataset_factory("basket_mv", conf)
dev = "cuda"

# Region of interest: around the point every camera looks at, sized by camera distance.
A, b = np.zeros((3, 3)), np.zeros(3)
for c in ds.data:
    c2w = c["extrinsics"]["c2w"]
    o, d = c2w[:, 3], c2w[:, 2] / np.linalg.norm(c2w[:, 2])
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ o
center = np.linalg.solve(A, b)
half = 0.6 * np.median([np.linalg.norm(c["extrinsics"]["c2w"][:, 3] - center) for c in ds.data])

ax = torch.arange(-half, half, args.voxel, device=dev)
gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3) + torch.tensor(center, device=dev, dtype=torch.float32)
votes = torch.zeros(len(pts), device=dev)
homo = torch.cat([pts, torch.ones(len(pts), 1, device=dev)], 1)

for i in range(len(ds)):
    s = ds[i]
    w2c = torch.as_tensor(np.asarray(s["cam_info"]["w2c"]), dtype=torch.float32, device=dev)
    K = torch.as_tensor(np.asarray(s["cam_info"]["K"]), dtype=torch.float32, device=dev)
    seg = s["seg"].to(dev).float()
    seg = seg[0] if seg.dim() == 3 else seg
    k = 5 if args.img_scale >= 0.5 else 3
    seg = F.max_pool2d(seg[None, None], k, 1, k // 2)[0, 0]  # 1-2 px tolerance
    H, W = seg.shape
    pc = homo @ w2c.T
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    u, v = uv[:, 0].long(), uv[:, 1].long()
    ok = (pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    hit = torch.zeros_like(votes)
    hit[ok] = seg[v[ok], u[ok]]
    votes += hit

keep = votes >= args.min_views_frac * len(ds)
occ = keep.reshape(gx.shape).float()
# surface = occupied voxels with at least one empty 6-neighbour
eroded = -F.max_pool3d(-occ[None, None], 3, 1, 1)[0, 0]
surface = (occ > 0) & (eroded == 0)
xyz = pts[surface.reshape(-1)].cpu().numpy()
print("center %s, half-size %.1f, voxel %.1f: %d occupied, %d surface points"
      % (center.round(1), half, args.voxel, int(keep.sum()), len(xyz)))

# train_dynamic_bind's load_obj negates x and y (Unreal -> COLMAP), so pre-negate here.
with open(args.out, "w") as f:
    for x, y, z in xyz:
        f.write("v %.4f %.4f %.4f\n" % (-x, -y, z))
print("wrote", args.out)
