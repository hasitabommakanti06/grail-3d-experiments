"""Run train_dynamic_bind.py with a different iteration count (the repo hard-codes 30k).

Set TRAIN_IMG_SCALE=<f> to train on images downscaled by f (the repo hard-codes 1.0).
Set SKIP_MASK_PRUNE=1 to skip the final prune-by-masks step.
usage: python train_bind_iters.py <iterations> <train_dynamic_bind.py args...>
"""
import os
import runpy
import sys

LIVESPLATS = "/scratch/hbommaka/code/basket-multiview/livesplats"
sys.path.insert(0, LIVESPLATS)
from utils.confs.optimizer import OptimizerConf  # noqa: E402

OptimizerConf.opt_confs["static"].iterations = int(sys.argv[1])

if os.environ.get("TRAIN_IMG_SCALE"):
    import datasets.basket_mv_dataset as bmv

    _init = bmv.BasketMVCamDataset.__init__

    def _scaled_init(self, conf):
        conf["img_scale_factor"] = float(os.environ["TRAIN_IMG_SCALE"])
        _init(self, conf)

    bmv.BasketMVCamDataset.__init__ = _scaled_init

if os.environ.get("SKIP_MASK_PRUNE"):
    # The repo's last step deletes every Gaussian that falls outside ANY camera's mask.
    # With occluded masks that deletes the whole player, so the occlusion test turns it off.
    from scene_hs import GaussianModel

    GaussianModel.filter_by_2d_masks = lambda self, dataset: None

script = os.path.join(LIVESPLATS, "scripts", "train_dynamic_bind.py")
sys.argv = [script] + sys.argv[2:]
os.chdir(LIVESPLATS)
runpy.run_path(script, run_name="__main__")
