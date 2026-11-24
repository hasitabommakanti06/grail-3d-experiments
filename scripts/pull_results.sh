#!/bin/bash
# Runs on the Mac. Copies results and input scenes from ece010 every 5 minutes until the
# queue finishes, because ece010 deletes /scratch files after 28 days.
# usage: caffeinate -i bash pull_results.sh
DEST="$HOME/Downloads/FALL 2026/GRAIL/Code/Server copy"
mkdir -p "$DEST"
while true; do
  rsync -a --prune-empty-dirs -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
    --exclude '*.pth' --exclude 'miniforge3*' --exclude 'envs' --exclude '.cache' --exclude 'data' --exclude 'code' \
    ece:/scratch/hbommaka/output ece:/scratch/hbommaka/logs "$DEST/" >> "$DEST/pull.log" 2>&1
  echo "$(date '+%F %T') pulled, rc=$?" >> "$DEST/pull.log"
  ssh -o BatchMode=yes -o ConnectTimeout=20 ece 'grep -q MASTER_DONE /scratch/hbommaka/logs/master.log' && break
  sleep 300
done
rsync -a -e "ssh -o BatchMode=yes" ece:/scratch/hbommaka/output ece:/scratch/hbommaka/logs "$DEST/" >> "$DEST/pull.log" 2>&1
echo "$(date '+%F %T') ALL_PULLED" >> "$DEST/pull.log"
