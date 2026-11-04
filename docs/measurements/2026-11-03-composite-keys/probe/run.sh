#!/bin/bash
# Kernel time of argsort2 and the two argsort on q32's real, uniform and shuffled columns.
# usage: run.sh CLICKBENCH_DATA_DIR SCRATCH_DIR, run from the repository root.
# The data directory holds the 1,000,000-row `WatchID.u64` and `ClientIP.u64` radix-image columns.
DATA=$1
SCRATCH=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(pwd)
export CARGO_INCREMENTAL=0
(cd "$HERE/q32-shape" && cargo build --release -q) || exit 1
mkdir -p "$SCRATCH/uniform" "$SCRATCH/shuffled"
# Prints the column shape (probe-q32-columns.txt) and writes the synthetic columns.
"$HERE/q32-shape/target/release/probe-q32" "$DATA" "$SCRATCH/uniform" "$SCRATCH/shuffled"
cargo build --release -j 3 --example acceptance_stats 2>&1 | tail -1
BIN=$ROOT/target/release/examples/acceptance_stats
export VERKEHRT_PROGRAMS=$HERE/programs
echo "start $(date +%T) load $(cat /proc/loadavg)"
for data in "$DATA:real" "$SCRATCH/uniform:uniform" "$SCRATCH/shuffled:shuffled"; do
  export VERKEHRT_CLICKBENCH=${data%%:*}
  echo "##### data: ${data##*:}"
  for q in c2 s1 s2; do
    echo "=== $q"
    $BIN $q 2>&1 | grep -E "^query|argsort"
  done
done
echo "end $(date +%T) load $(cat /proc/loadavg)"
