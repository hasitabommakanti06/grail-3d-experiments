"""Sanity-check that camera poses agree with the masks.

Finds the 3D point closest to every camera's optical axis (what the rig looks at),
projects it into each image with the loader's own K / w2c, and reports whether
it lands inside that camera's mask.
usage: python check_geometry.py <scene dir with cameras/ and masks/> [rgb_source_path]
"""
import sys

import numpy as np

sys.path.insert(0, "/scratch/hbommaka/code/basket-multiview/livesplats")
import datasets.dataset_factory as dataset_factory  # noqa: E402

conf = dict(dataset_factory.DATASET_CONF)
conf.update(data_path=sys.argv[1], timestamp=0, spatial_scale_factor=1, img_scale_factor=1.0,
            load_depth=False, require_all_img_data_available=False,
            dataset_caching=False, first_k_cams=-1, erosion_kernel_size=0)
if len(sys.argv) > 2:
    conf["rgb_source_path"] = sys.argv[2]
ds = dataset_factory.dataset_factory("basket_mv", conf)

A, b = np.zeros((3, 3)), np.zeros(3)
for c in ds.data:
    c2w = c["extrinsics"]["c2w"]
    o, d = c2w[:, 3], c2w[:, 2] / np.linalg.norm(c2w[:, 2])
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ o
X = np.linalg.solve(A, b)
print("rig look-at point:", X.round(2))

inside = 0
for i in range(len(ds)):
    s = ds[i]
    w2c, K = np.asarray(s["cam_info"]["w2c"]), np.asarray(s["cam_info"]["K"])
    pc = w2c @ np.append(X, 1)
    u, v = (K @ pc)[:2] / pc[2]
    seg = s["seg"].cpu().numpy()
    seg = seg[0] if seg.ndim == 3 else seg
    H, W = seg.shape
    ok = 0 <= int(v) < H and 0 <= int(u) < W and seg[int(v), int(u)] > 0
    inside += ok
    if i < 6 or not ok:
        print("cam %2d id %2d: depth %7.1f  pixel (%6.0f, %6.0f)  mask cover %.3f  in-mask %s"
              % (i, s["id"], pc[2], u, v, seg.mean(), ok))
print("%d / %d cameras see the look-at point inside the mask" % (inside, len(ds)))
