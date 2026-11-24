# 3D player reconstruction experiments (GRAIL x YinzCam)

Scripts and results from my experiments on 3D reconstruction of a player from a camera array.
This is working code, not cleaned up.

## What this is

I use the LiveSplats code from [humansensinglab/basket-multiview](https://github.com/humansensinglab/basket-multiview)
(commit `35105cb`) and its synthetic 40-camera dataset. I reconstruct one player from one frame with
3D Gaussian splatting, then score the result with PSNR on 5 cameras that were not used for training.
Each experiment changes one thing (number of cameras, calibration error, player size, camera placement,
time sync, training steps, occlusion, mask errors).

The LiveSplats code itself is not copied here. Clone it from the link above. I changed two small things in it
to get it to run on the released data:

- `train_dynamic_bind` did not handle the `default` renderer
- `DefaultLoss` expected a `motion_info` input that the single-frame data does not have

The released dataset does not include the player meshes, so `scripts/carve.py` builds a starting shape
from the player silhouettes (a visual hull) and training starts from that.

## Layout

- `scripts/` everything I wrote
  - `setup_env.sh` builds the conda environment (torch 2.3.0, cu121)
  - `share.py` downloads the parts of the dataset I used
  - `carve.py` starting shape from the masks
  - `camera_count_exp.py`, `exp_variants.py`, `mask_variants.py` build the scene for each run
  - `train_bind_iters.py` runs LiveSplats training for a set number of steps
  - `run_camera_count.sh`, `run_overnight.sh`, `run_new.sh` the three queues (64 runs in total)
  - `plot_*.py`, `summarize_overnight.py` charts and tables
- `results/` charts, example images, `Results summary.md` and `All numbers.csv`
- `camera_split.json` which cameras are for training and which 5 are held out

Paths in the scripts point to `/scratch/hbommaka` on the ECE server (one Tesla T4), so they need editing
to run anywhere else. One run of 10,000 steps takes about 7 minutes on the T4.

## License and credit

The LiveSplats code and the BASKET-Multiview dataset are from the CMU Human Sensing Lab
("Echoes of the Coliseum", ACM TOG 2025) and are released under CC BY-NC 4.0, so non-commercial use only.
The example images in `results/` are renders of that dataset. A copy of the license is in
`LICENSE-upstream-CC-BY-NC-4.0.txt`.
