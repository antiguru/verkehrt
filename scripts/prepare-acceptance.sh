#!/usr/bin/env bash
# Prepares the acceptance data: corgi's ClickBench and TPC-H inputs and DuckDB's expected answers.
#
# Usage: scripts/prepare-acceptance.sh DATA_DIR
#
# Produces under DATA_DIR:
#   corgi-wip/      clone of corgi's repository at the pinned revision
#   hits_0.parquet  first ClickBench partition
#   clickbench/     prepare.py output (*.u64, *.ends, *.bytes, expected/*.txt, duckdb.tsv)
#   tpch/           prepare.py output at scale factor 0.2
# duckdb is pinned below 2 because prepare.py calls Relation.fetchone(), which duckdb 2.0 removed.
# Every step skips work that is already done. Needs git, curl and uv.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 DATA_DIR" >&2; exit 2; }
mkdir -p "$1"
DATA_DIR=$(cd "$1" && pwd)

REPO=https://github.com/frankmcsherry/WIP
REV=2751e04e8501d47b33219be23049bf4afa7a4428
HITS=https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet

if [ ! -d "$DATA_DIR/corgi-wip" ]; then
    git clone "$REPO" "$DATA_DIR/corgi-wip"
fi
git -C "$DATA_DIR/corgi-wip" checkout --quiet "$REV"

if [ ! -f "$DATA_DIR/hits_0.parquet" ]; then
    # Download to a temporary name so an interrupted transfer is not mistaken for the file.
    curl -fL --retry 3 -o "$DATA_DIR/hits_0.parquet.part" "$HITS"
    mv "$DATA_DIR/hits_0.parquet.part" "$DATA_DIR/hits_0.parquet"
fi

# prepare.py reads the queries relative to its own location, so the working directory does not matter.
# duckdb.tsv is written last, so its presence marks a finished run.
if [ ! -f "$DATA_DIR/clickbench/duckdb.tsv" ]; then
    uv run --with "duckdb<2" --with numpy python3 \
        "$DATA_DIR/corgi-wip/corgi/benches/clickbench/prepare.py" \
        "$DATA_DIR/clickbench" "$DATA_DIR/hits_0.parquet"
fi
if [ ! -f "$DATA_DIR/tpch/duckdb.tsv" ]; then
    uv run --with "duckdb<2" --with numpy python3 \
        "$DATA_DIR/corgi-wip/corgi/benches/tpch/prepare.py" \
        "$DATA_DIR/tpch" 0.2
fi

uv run --with "duckdb<2" python3 -c 'import duckdb; print("duckdb", duckdb.__version__)'
