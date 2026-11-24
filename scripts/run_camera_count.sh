#!/bin/bash
# Camera-count experiment: same player, fewer and fewer training cameras,
# always scored on the same 5 held-out cameras.
set -o pipefail
source /scratch/hbommaka/env_local.sh
T=/scratch/hbommaka/tools
SRC=/scratch/hbommaka/data/assets/players_A_pose/male
RGB=$SRC/rgb/clothing_black/skin_color_1
EXP=/scratch/hbommaka/output/exp_camera_count
ITERS=${ITERS:-10000}
mkdir -p $EXP
[ -f $EXP/splits.json ] || python $T/camera_count_exp.py prepare --src $SRC --out $EXP --n_test 5 --subsets 35,20,10,6,4

for N in 35 20 10 6 4; do
  S=$EXP/$(printf "n%02d" $N)
  [ -f $S/eval.json ] && { echo "== $N cams: done already"; continue; }
  echo "== $N cams: carve"
  python $T/carve.py $S $S/hull.obj --rgb_source_path $RGB || { echo "CARVE_FAILED $N"; continue; }
  echo "== $N cams: train ($ITERS iters)"
  START=$(date +%s)
  python $T/train_bind_iters.py $ITERS --source_path $S --rgb_source_path $RGB --first_frame_obj $S/hull.obj \
      --model_path $S/model --dataset_type basket_mv > $S/train.log 2>&1 || { echo "TRAIN_FAILED $N"; tail -5 $S/train.log; continue; }
  echo "train_seconds $(( $(date +%s) - START ))" > $S/timing.txt
  echo "== $N cams: evaluate"
  python $T/camera_count_exp.py evaluate --test_scene $EXP/test --rgb_source_path $RGB \
      --ply $S/model/0/checkpoints/0000.ply --out $S/eval.json || echo "EVAL_FAILED $N"
done
echo ALL_DONE
