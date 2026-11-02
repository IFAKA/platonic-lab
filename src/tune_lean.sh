#!/bin/sh
# PRE-REGISTERED lean tune (written before any tune run):
#  3 methods x LR {2e-4, 5e-4, 1.25e-3}, seed 0, 1000 steps, same data/batch. Score = final mkNN on test images.
#  Rule: if a method's best LR is 5e-4, or beats 5e-4 by < 0.01 mkNN, the main table stands for it.
#        If a method's best LR beats 5e-4 by >= 0.01, rerun that method at that LR for 3000 steps, seeds 0-2, and replace it.
# Waits for the main runs (ALLDONE in results/run.log) so the two jobs do not slow each other.
cd "$(dirname "$0")/.."
until grep -q ALLDONE results/run.log 2>/dev/null; do sleep 20; done
for lr in 2e-4 5e-4 1.25e-3; do for o in good bad ntp; do
  LR=$lr OUT=tune .venv/bin/python src/lab.py data/coco_10000.pt $o 0 1000 2>&1 | grep -v -i -E "warn|redirect" >> tune.log
done; done
echo TUNEDONE >> tune.log
