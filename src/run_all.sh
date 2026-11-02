#!/bin/sh
# usage: run_all.sh DATA.pt  -> runs 3 objectives x 3 seeds sequentially (one GPU), logs to results/run.log
cd "$(dirname "$0")/.." && mkdir -p results
for s in 0 1 2; do for o in good bad ntp; do
  .venv/bin/python src/lab.py "$1" $o $s 3000 2>&1 | grep -v -i -E "warn|redirect" >> results/run.log
done; done
echo ALLDONE >> results/run.log
