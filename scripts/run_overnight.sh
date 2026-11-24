#!/bin/bash
# Overnight experiment queue (restartable: finished runs are skipped).
# Everything lives on /scratch so it keeps working after the AFS token expires.
set -o pipefail
source /scratch/hbommaka/env_local.sh
T=/scratch/hbommaka/tools
OUT=/scratch/hbommaka/output/overnight
APOSE=/scratch/hbommaka/data/assets/players_A_pose/male
APOSE_RGB=$APOSE/rgb/clothing_black/skin_color_1
CC=/scratch/hbommaka/output/exp_camera_count
TEST=$CC/test
N20=$(python -c "import json; print(','.join(map(str, json.load(open('$CC/splits.json'))['splits']['n20'])))")
TEST_IDS=$(python -c "import json; print(','.join(map(str, json.load(open('$CC/splits.json'))['splits']['test'])))")
mkdir -p $OUT

# run GROUP NAME SCENE RGB TEST_SCENE TEST_RGB ITERS SCALE
run() {
  local G=$1 N=$2 S=$3 RGB=$4 TS=$5 TRGB=$6 IT=$7 SC=$8
  local D=$OUT/$G/$N
  mkdir -p $D
  [ -f $D/eval.json ] && { echo "== $G/$N done already"; return; }
  echo "== $G/$N  start $(date +%T)"
  python $T/carve.py $S $D/hull.obj --rgb_source_path $RGB --img_scale $SC > $D/carve.log 2>&1 || { echo "CARVE_FAILED $G/$N"; tail -3 $D/carve.log; return; }
  local START=$(date +%s)
  TRAIN_IMG_SCALE=$SC python $T/train_bind_iters.py $IT --source_path $S --rgb_source_path $RGB \
      --first_frame_obj $D/hull.obj --model_path $D/model --dataset_type basket_mv > $D/train.log 2>&1 \
      || { echo "TRAIN_FAILED $G/$N"; tail -4 $D/train.log; return; }
  echo "{\"train_seconds\": $(( $(date +%s) - START )), \"iters\": $IT, \"img_scale\": $SC}" > $D/run.json
  python $T/camera_count_exp.py evaluate --test_scene $TS --rgb_source_path $TRGB \
      --ply $D/model/0/checkpoints/0000.ply --out $D/eval.json > $D/eval.log 2>&1 || { echo "EVAL_FAILED $G/$N"; tail -3 $D/eval.log; return; }
  echo "RESULT $G/$N $(grep -m1 psnr_box $D/eval.json | tr -d ' ,')"
}

# cams GROUP NAME [exp_variants cams flags...]  -> builds a 20-camera A-pose scene, then runs it
cams() {
  local G=$1 N=$2; shift 2
  local S=$OUT/$G/$N/scene
  [ -d $S ] || python $T/exp_variants.py cams --src $APOSE --out $S --ids $N20 "$@" > /dev/null
  run $G $N $S $APOSE_RGB $TEST $APOSE_RGB 10000 1.0
}

# --- 1. run-to-run noise --------------------------------------------------
cams baseline r2 --seed 2

# --- 2. calibration: rotation error (degrees, every camera) ----------------
for X in 0.1 0.5 1.0 0.25 0.05; do cams calib_rot deg$X --rot_deg $X --seed 1; done

# --- 3. resolution (fewer pixels on the player) ----------------------------
for SC in 0.25 0.0625 0.125 0.5; do
  run resolution scale$SC $CC/n20 $APOSE_RGB $TEST $APOSE_RGB 10000 $SC
done

# --- 4. camera layout (10 cameras each) ------------------------------------
for R in upper side notop ring0; do
  S=$OUT/layout/$R/scene
  if [ ! -d $S ]; then
    IDS=$(python $T/exp_variants.py layout --splits $CC/splits.json --rule $R --n 10)
    python $T/exp_variants.py cams --src $APOSE --out $S --ids $IDS > /dev/null
    echo "layout $R cameras: $IDS"
  fi
  run layout $R $S $APOSE_RGB $TEST $APOSE_RGB 10000 1.0
done

# --- 5. time sync (running player; half the training cameras lag) -----------
RB=/scratch/hbommaka/data/development/running_black
RBS=$RB/male
RB_RGB=$RBS/rgb/skin_color_1
T0=20
LATE=$(python -c "ids='$N20'.split(','); print(','.join(ids[1::2]))")
SYNC_TEST=$OUT/sync/test_scene
[ -d $SYNC_TEST ] || python $T/exp_variants.py sync --seq $RBS --rgb $RB_RGB --cams $RB/cameras --out $SYNC_TEST --ids $TEST_IDS --t0 $T0 > /dev/null
for K in 0 1 2 4; do
  S=$OUT/sync/lag$K/scene
  [ -d $S ] || python $T/exp_variants.py sync --seq $RBS --rgb $RB_RGB --cams $RB/cameras --out $S --ids $N20 --late_ids $LATE --t0 $T0 --offset $K > /dev/null
  run sync lag$K $S $S/rgb $SYNC_TEST $SYNC_TEST/rgb 10000 1.0
done

# --- 6. compute budget (training steps) -------------------------------------
for IT in 500 2000 5000 30000; do
  run iterations it$IT $CC/n20 $APOSE_RGB $TEST $APOSE_RGB $IT 1.0
done

# --- 7. calibration: camera position error (scene units = cm) ----------------
for X in 1 5 0.5 2; do cams calib_trans cm$X --trans $X --seed 1; done

# --- 8. calibration: focal length (zoom) error -------------------------------
for X in 1 2 0.5; do cams calib_focal pct$X --focal_pct $X --seed 1; done

# --- 9. one more noise repeat ----------------------------------------------
cams baseline r3 --seed 3

python $T/summarize_overnight.py $OUT > $OUT/summary.txt 2>&1
echo ALL_DONE $(date +%T)
