#!/bin/bash
# New experiments (Nov 24): mask quality, occlusion, finer time-sync test.
# Same recipe as run_overnight.sh: 20 training cameras, 10k steps, scored on the 5 held-out cameras.
# Restartable: finished runs are skipped.
set -o pipefail
source /scratch/hbommaka/env_local.sh
T=/scratch/hbommaka/tools
OUT=/scratch/hbommaka/output/new
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
  python $T/carve.py $S $D/hull.obj --rgb_source_path $RGB --img_scale $SC --min_views_frac ${CARVE_FRAC:-0.95} > $D/carve.log 2>&1 || { echo "CARVE_FAILED $G/$N"; tail -3 $D/carve.log; return; }
  local START=$(date +%s)
  TRAIN_IMG_SCALE=$SC python $T/train_bind_iters.py $IT --source_path $S --rgb_source_path $RGB \
      --first_frame_obj $D/hull.obj --model_path $D/model --dataset_type basket_mv > $D/train.log 2>&1 \
      || { echo "TRAIN_FAILED $G/$N"; tail -4 $D/train.log; return; }
  echo "{\"train_seconds\": $(( $(date +%s) - START )), \"iters\": $IT, \"img_scale\": $SC}" > $D/run.json
  python $T/camera_count_exp.py evaluate --test_scene $TS --rgb_source_path $TRGB \
      --ply $D/model/0/checkpoints/0000.ply --out $D/eval.json > $D/eval.log 2>&1 || { echo "EVAL_FAILED $G/$N"; tail -3 $D/eval.log; return; }
  echo "RESULT $G/$N $(grep -m1 psnr_box $D/eval.json | tr -d ' ,')"
}

# mask GROUP NAME [mask_variants scene flags...] -> degrade the masks of the 20-camera A-pose scene, then run it
mask() {
  local G=$1 N=$2; shift 2
  local S=$OUT/$G/$N/scene
  [ -f $S/variant.json ] || python $T/mask_variants.py scene --src $CC/n20 --mask_src $APOSE/masks --out $S "$@" || { echo "SCENE_FAILED $G/$N"; return; }
  run $G $N $S $APOSE_RGB $TEST $APOSE_RGB 10000 1.0
}

# --- 1. occlusion: part of the player hidden in some cameras -----------------
# As released, the code can't run this: its final step prunes any Gaussian outside any camera's
# mask, which removes the whole player (crash log kept in occlusion_released_code/). So these runs
# skip that step, and the starting hull only needs a voxel to be seen by the cameras that could see it.
CARVE_FRAC=0.54 SKIP_MASK_PRUNE=1 mask occlusion cams50_cover50 --mode occlude --frac 0.5 --cover 0.5
CARVE_FRAC=0.64 SKIP_MASK_PRUNE=1 mask occlusion cams25_cover50 --mode occlude --frac 0.25 --cover 0.5
CARVE_FRAC=0.43 SKIP_MASK_PRUNE=1 mask occlusion cams75_cover50 --mode occlude --frac 0.75 --cover 0.5
CARVE_FRAC=0.32 SKIP_MASK_PRUNE=1 mask occlusion cams100_cover50 --mode occlude --frac 1.0 --cover 0.5
CARVE_FRAC=0.54 SKIP_MASK_PRUNE=1 mask occlusion cams50_cover25 --mode occlude --frac 0.5 --cover 0.25
CARVE_FRAC=0.32 SKIP_MASK_PRUNE=1 mask occlusion cams50_cover75 --mode occlude --frac 0.5 --cover 0.75
# control: clean masks, prune step skipped, to separate the effect of skipping it
SKIP_MASK_PRUNE=1 run occlusion cams0_noprune $CC/n20 $APOSE_RGB $TEST $APOSE_RGB 10000 1.0

# --- 2. mask quality ----------------------------------------------------------
for X in 5 2 10; do mask mask_erode px$X --mode erode --px $X; done
for X in 5 2 10; do mask mask_dilate px$X --mode dilate --px $X; done
for X in 4 8; do mask mask_jitter px$X --mode jitter --px $X; done
mask mask_holes r40 --mode holes --px 40 --n_holes 3

# --- 3. time sync vs. how fast the player is moving ---------------------------
# Frames are 33 ms apart, so we can't shift a camera by less than one frame. Instead we
# use moments of the jog where the body moves less between frames, and measure the motion.
RB=/scratch/hbommaka/data/development/running_black
RBS=$RB/male
RB_RGB=$RBS/rgb/skin_color_1
LATE=$(python -c "ids='$N20'.split(','); print(','.join(ids[1::2]))")
mkdir -p $OUT/sync_motion
[ -f $OUT/sync_motion/motion.json ] || python $T/mask_variants.py motion --masks $RBS/masks --ids $LATE --out $OUT/sync_motion/motion.json
T0S=$(python -c "import json; print(' '.join(map(str, json.load(open('$OUT/sync_motion/motion.json'))['picked_t0'])))")
echo "sync frames (slowest to fastest): $T0S"
sync_scene() {  # sync_scene OUTDIR IDS LATE_IDS T0 OFFSET
  [ -d $1 ] || python $T/exp_variants.py sync --seq $RBS --rgb $RB_RGB --cams $RB/cameras --out $1 --ids $2 ${3:+--late_ids $3} --t0 $4 --offset $5 > /dev/null
}
for T0 in $T0S; do
  TS=$OUT/sync_motion/test_t$T0
  sync_scene $TS $TEST_IDS "" $T0 0
  for K in 1 0; do
    S=$OUT/sync_motion/t${T0}_lag$K/scene
    sync_scene $S $N20 $LATE $T0 $K
    run sync_motion t${T0}_lag$K $S $S/rgb $TS $TS/rgb 10000 1.0
  done
done

# --- 4. time sync: how many cameras are off (1 frame late, same frame as the October test) ---
T0=20
TS=$OUT/sync_count/test_scene
sync_scene $TS $TEST_IDS "" $T0 0
for NL in 1 5; do
  L=$(python -c "print(','.join('$LATE'.split(',')[:$NL]))")
  S=$OUT/sync_count/late$NL/scene
  sync_scene $S $N20 $L $T0 1
  run sync_count late$NL $S $S/rgb $TS $TS/rgb 10000 1.0
done

# --- 5. same sync test on the slowest moment of the dribbling clip (least motion we have) ---
RB=/scratch/hbommaka/data/development/dribbling_black
RBS=$RB/male
RB_RGB=$RBS/rgb/skin_color_1
[ -f $OUT/sync_motion/motion_dribbling.json ] || python $T/mask_variants.py motion --masks $RBS/masks --ids $LATE --out $OUT/sync_motion/motion_dribbling.json > /dev/null
T0=$(python -c "import json; print(json.load(open('$OUT/sync_motion/motion_dribbling.json'))['picked_t0'][0])")
TS=$OUT/sync_motion/test_drib_t$T0
sync_scene $TS $TEST_IDS "" $T0 0
for K in 1 0; do
  S=$OUT/sync_motion/drib_t${T0}_lag$K/scene
  sync_scene $S $N20 $LATE $T0 $K
  run sync_motion drib_t${T0}_lag$K $S $S/rgb $TS $TS/rgb 10000 1.0
done

python $T/summarize_overnight.py $OUT > $OUT/summary.txt 2>&1
echo NEW_DONE $(date +%T)
