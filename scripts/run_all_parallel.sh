#!/usr/bin/env bash
# Run "grzybki optimize" + "grzybki visualize" for every dataset in data/instances,
# in parallel, one seed per dataset.
#
# Usage:
#   bash scripts/run_all_parallel.sh            # run all datasets
#   bash scripts/run_all_parallel.sh --force    # re-run even if a run dir is already complete
#
# Budgets (the pure-Python decoder is quadratic in the number of points, so the
# largest instances get scaled-down budgets to keep wall time reasonable):
#   n <=  100 : default K = n-2, 20000 evals
#   n ==  500 : default K = n-2,  1500 evals   (~17 min)
#   n == 1000 : K = 200,          1000 evals   (~19 min)
#   n == 5000 : K = 200,           120 evals   (~60 min, ~3 GB RAM per eval)
#
# Outputs:   runs/<instance>_seed42            (run artifacts)
#            runs/<instance>_seed42/figures    (convergence, network, animation, report)
# Logs:      runs/_parallel_logs/<instance>.log
# Status:    runs/_parallel_logs/status.tsv    (name<TAB>ok|skipped|optimize_failed|visualize_failed)
#
# Concurrency plan (12 cores / ~10 GB free RAM):
#   5000-size: 2 at a time (~3 GB each)
#   1000-size then 500-size: 2 at a time (sequential pools)
#   small (100/50/30/15): 4 at a time
# Expected total wall time: roughly 2 hours (the 5000-size pool dominates).

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRZYBKI="$ROOT/.venv/bin/grzybki"
LOGS="$ROOT/runs/_parallel_logs"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ ! -x "$GRZYBKI" ]; then
    echo "error: $GRZYBKI not found (create the project venv first)" >&2
    exit 1
fi

mkdir -p "$LOGS"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

families="clustered grid poland polygon ring uniform"

args_for() {
    case "$1" in
        *_5000) echo "--seed 42 --max-evals 120 --K 200 --snapshot-interval 40" ;;
        *_1000) echo "--seed 42 --max-evals 1000 --K 200 --snapshot-interval 100" ;;
        *_500)  echo "--seed 42 --max-evals 1500 --snapshot-interval 100" ;;
        *)      echo "--seed 42 --max-evals 20000 --snapshot-interval 100" ;;
    esac
}

worker() { # worker <instance_name> <optimize args>
    local name="$1" args="$2"
    local run_dir="$ROOT/runs/${name}_seed42"
    local log="$LOGS/${name}.log"
    local rc
    if [ "$FORCE" -eq 0 ] && [ -f "$run_dir/summary.json" ] && [ -f "$run_dir/figures/report.html" ]; then
        printf '%s\tskipped\n' "$name" >> "$LOGS/status.tsv"
        return 0
    fi
    echo "=== OPTIMIZE start $(date -Is) args: $args" >> "$log"
    "$GRZYBKI" optimize "$ROOT/data/instances/${name}.json" $args --out "$run_dir" >> "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "=== OPTIMIZE FAILED rc=$rc $(date -Is)" >> "$log"
        printf '%s\toptimize_failed\n' "$name" >> "$LOGS/status.tsv"
        return 1
    fi
    echo "=== VISUALIZE start $(date -Is)" >> "$log"
    "$GRZYBKI" visualize "$run_dir" --out "$run_dir/figures" >> "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "=== VISUALIZE FAILED rc=$rc $(date -Is)" >> "$log"
        printf '%s\tvisualize_failed\n' "$name" >> "$LOGS/status.tsv"
        return 1
    fi
    echo "=== DONE $(date -Is)" >> "$log"
    printf '%s\tok\n' "$name" >> "$LOGS/status.tsv"
}

pool() { # pool <size list> <concurrency>
    local sizes="$1" conc="$2"
    local running=0 size fam name
    for size in $sizes; do
        for fam in $families; do
            name="${fam}_${size}"
            while [ "$running" -ge "$conc" ]; do
                wait -n
                running=$((running - 1))
            done
            worker "$name" "$(args_for "$name")" &
            running=$((running + 1))
        done
    done
    wait
}

echo "start $(date -Is) (logs: $LOGS)"

pool "5000" 2 &                  # ~3 GB peak per eval -> 2 at a time
P1=$!
( pool "1000" 2; pool "500" 2 ) &
P2=$!
pool "100 50 30 15" 4 &
P3=$!
wait "$P1" "$P2" "$P3"

ok=$(grep -c $'\tok$' "$LOGS/status.tsv" || true)
skipped=$(grep -c $'\tskipped$' "$LOGS/status.tsv" || true)
failed=$(grep -c 'failed$' "$LOGS/status.tsv" || true)
echo "done $(date -Is): ok=$ok skipped=$skipped failed=$failed"
echo "status table: $LOGS/status.tsv"
[ "$failed" -eq 0 ] || exit 1
