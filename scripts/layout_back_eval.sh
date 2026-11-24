#!/bin/bash
# Score every layout model on 3 cameras behind the player that none of them trained on.
source /scratch/hbommaka/env_local.sh
T=/scratch/hbommaka/tools
APOSE=/scratch/hbommaka/data/assets/players_A_pose/male
RGB=$APOSE/rgb/clothing_black/skin_color_1
OUT=/scratch/hbommaka/output/overnight
BACK=$OUT/layout/back_test_scene
[ -d $BACK ] || python $T/exp_variants.py cams --src $APOSE --out $BACK --ids 9,10,18
for M in /scratch/hbommaka/output/exp_camera_count/n10 $OUT/layout/upper $OUT/layout/notop $OUT/layout/side $OUT/layout/ring0; do
  P=$(ls $M/model/0/checkpoints/0000.ply 2>/dev/null || ls $M/0/checkpoints/0000.ply 2>/dev/null)
  [ -z "$P" ] && P=$M/model/0/checkpoints/0000.ply
  python $T/camera_count_exp.py evaluate --test_scene $BACK --rgb_source_path $RGB --ply $P --out $M/eval_back.json 2>&1 | grep mean
done
echo BACK_DONE
