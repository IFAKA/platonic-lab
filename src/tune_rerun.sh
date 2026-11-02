#!/bin/sh
# Follow-up to tune_lean.sh, written after seeing it and BEFORE these runs:
#  The lean rule fired for all 3 methods (best LR 1.25e-3 beat 5e-4 by >= 0.01), and 1.25e-3 was the edge of the grid.
#  Step 1: probe LR 3e-3 the same way (seed 0, 1000 steps). Best LR per method by mkNN over all tune runs.
#  Step 2: rerun each method at its best LR, 3000 steps -> results_tuned/. Seeds outer: seed 0 for all three first.
#  Sequential stop: if after seed 0 good leads both others by >= 0.08 mkNN (main table: seed spread <= 0.034), stop;
#  else run seeds 1 and 2.
cd "$(dirname "$0")/.."
while pgrep -q -f "lab.py data/coco_10000.pt bad 0 1000" ; do sleep 10; done  # own probe started by the first version
for o in good bad ntp; do
  [ -f tune/${o}_s0_lr0.003.json ] || LR=3e-3 OUT=tune .venv/bin/python src/lab.py data/coco_10000.pt $o 0 1000 2>&1 | grep -v -i -E "warn|redirect" >> tune.log
done
best() { .venv/bin/python -c "import json,glob;r={json.load(open(f))['lr']:json.load(open(f))['log'][-1]['mknn'] for f in glob.glob('tune/$1_s0_lr*.json')};print(max(r,key=r.get))"; }
for s in 0 1 2; do
  for o in good bad ntp; do
    LR=$(best $o) OUT=results_tuned .venv/bin/python src/lab.py data/coco_10000.pt $o $s 3000 2>&1 | grep -v -i -E "warn|redirect" >> tune.log
  done
  echo "SEED$s DONE" >> tune.log
  if [ $s = 0 ]; then
    .venv/bin/python -c "
import json,glob
m={o:json.load(open(glob.glob(f'results_tuned/{o}_s0_*.json')[0]))['log'][-1]['mknn'] for o in ['good','bad','ntp']}
exit(0 if m['good']-max(m['bad'],m['ntp'])>=0.08 else 1)" && { echo "STOP: gap clear after seed 0" >> tune.log; break; }
  fi
done
echo RERUNDONE >> tune.log
