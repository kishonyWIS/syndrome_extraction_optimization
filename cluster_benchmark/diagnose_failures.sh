#!/bin/bash
# Run on the cluster (where log/ and error/ live) to summarize why jobs EXIT.
# Usage: bash syndrome_extraction_optimization/cluster_benchmark/diagnose_failures.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERR_DIR="${SCRIPT_DIR}/error"
LOG_DIR="${SCRIPT_DIR}/log"

echo "=== Failure diagnosis (error/*.err) ==="
echo ""

if [ ! -d "$ERR_DIR" ]; then
  echo "No error dir: $ERR_DIR"
  exit 0
fi

n_err=$(find "$ERR_DIR" -name 'benchmark-*.err' 2>/dev/null | wc -l)
echo "Total .err files: $n_err"
echo ""

# Sample up to 5 full files and summarize patterns
echo "--- Common patterns (grep over all .err) ---"
for pattern in "Killed\|Out of memory\|MemoryError\|OOM\|Exceeded job memory limit" \
              "Illegal instruction\|illegal operand\|SIGILL" \
              "BENCHMARK_JOB_FAILED" \
              "ModuleNotFoundError\|ImportError" \
              "No such file\|FileNotFoundError" \
              "sinter\|stim" \
              "Traceback\|Error:"; do
  label=$(echo "$pattern" | sed 's/\\|/ or /g')
  count=$(grep -l -E "$pattern" "$ERR_DIR"/benchmark-*.err 2>/dev/null | wc -l || true)
  printf "  %-50s %s\n" "$label" "$count"
done

echo ""
echo "--- Sample of last 15 lines from 3 recent .err files ---"
for f in $(find "$ERR_DIR" -name 'benchmark-*.err' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2-); do
  echo ">>> $f"
  tail -15 "$f" 2>/dev/null || true
  echo ""
done

echo "To inspect a specific job: cat $ERR_DIR/benchmark-<JOBID>.err"
