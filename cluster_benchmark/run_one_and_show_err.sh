#!/bin/bash
# Submit ONE benchmark job (d=9 midout chunk=0), wait for it, then print its .err.
# Run from repo root on the cluster: bash syndrome_extraction_optimization/cluster_benchmark/run_one_and_show_err.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLUSTER_DIR="$SCRIPT_DIR"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# One job's config (matches submit_benchmark_sweep.sh)
noise_model=tqec_uniform_depolarizing
distance=9
ROUNDS=9
error_rate=0.001
circuit_type=midout
decoder=tesseract
chunk=0
RUN_ID=debug_one_job
LSF_MEM_CUR=1200
LSF_QUEUE="${LSF_QUEUE:-berg}"
LSF_NCPUS=4
LSF_WALLTIME="72:00"
LSF_LOG_DIR="$CLUSTER_DIR/log"
LSF_ERR_DIR="$CLUSTER_DIR/error"
N_SHOTS_PER_CHUNK=2000
MAX_ERRORS_PER_CHUNK=3
NUM_WORKERS=4
SCHEDULE_CSV="${SCHEDULE_CSV:-$BENCHMARK_DIR/results/zero_collision_schedules.csv}"
SCHEDULE_LINE="  \\"

OUT_BASE="$CLUSTER_DIR/output/$noise_model"
OUTPUT_FILE="$OUT_BASE/result_${distance}_${error_rate}_${circuit_type}_${decoder}_chunk${chunk}_${RUN_ID}.csv"
mkdir -p "$OUT_BASE" "$LSF_LOG_DIR" "$LSF_ERR_DIR"

TEMP_SCRIPT="$CLUSTER_DIR/job_scripts/run_one_debug_$$.sh"
mkdir -p "$CLUSTER_DIR/job_scripts"

cat > "$TEMP_SCRIPT" << 'JOBEOF'
#!/bin/bash
#BSUB -q berg
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model!=AMD_EPYC]"
#BSUB -R "rusage[mem=1200]"
#BSUB -M 1200
#BSUB -W 72:00
#BSUB -o __LSF_LOG_DIR__/benchmark-%J.out
#BSUB -e __LSF_ERR_DIR__/benchmark-%J.err

set -euo pipefail
_fail() { echo "BENCHMARK_JOB_FAILED: exit=$1 (d=9 midout chunk=0)" >&2; exit "$1"; }
trap '_fail $?' ERR

cd "__REPO_ROOT__"
module load Python/3.11.3-GCCcore-12.3.0
export LD_LIBRARY_PATH="${EBROOTPYTHON:-/apps/easybd/easybuild/amd/software/Python/3.11.3-GCCcore-12.3.0}/lib:${LD_LIBRARY_PATH:-}"
source "__REPO_ROOT__/venv/bin/activate"
export NPY_DISABLE_CPU_FEATURES="X86_V4,X86_V3"

rm -f "__OUTPUT_FILE__"
CIRCUIT_FILE="${TMPDIR:-/tmp}/circuit_$$.stim"
SINTER_CSV="${TMPDIR:-/tmp}/sinter_$$.csv"

python3 "__REPO_ROOT__/syndrome_extraction_optimization/cluster_benchmark/build_circuit_only.py" \
  --distance 9 --error-rate 0.001 --noise-model tqec_uniform_depolarizing \
  --circuit-type midout --rounds 9 \
  --output "$CIRCUIT_FILE"

export PYTHONPATH="__REPO_ROOT__/syndrome_extraction_optimization:__REPO_ROOT__"
sinter collect \
  --circuits "$CIRCUIT_FILE" \
  --decoders tesseract \
  --custom_decoders_module_function "benchmark_circuits:get_custom_decoders" \
  --max_shots 2000 \
  --max_errors 3 \
  --processes 4 \
  --save_resume_filepath "$SINTER_CSV"

python3 "__REPO_ROOT__/syndrome_extraction_optimization/cluster_benchmark/sinter_csv_to_benchmark_row.py" \
  "$SINTER_CSV" -o "__OUTPUT_FILE__" \
  --distance 9 --rounds 9 --p-cnot 0.001 \
  --circuit-type midout --decoder tesseract --noise-model tqec_uniform_depolarizing \
  --n-shots 2000

rm -f "$CIRCUIT_FILE" "$SINTER_CSV"
JOBEOF

sed -i "s|__REPO_ROOT__|$REPO_ROOT|g; s|__LSF_LOG_DIR__|$LSF_LOG_DIR|g; s|__LSF_ERR_DIR__|$LSF_ERR_DIR|g; s|__OUTPUT_FILE__|$OUTPUT_FILE|g" "$TEMP_SCRIPT"
chmod +x "$TEMP_SCRIPT"

echo "Submitting one job (d=9 midout chunk=0)..."
out=$(bsub -J "sweep_d9_mido_c0" "$TEMP_SCRIPT" 2>&1)
echo "$out"
JOB_ID=$(echo "$out" | grep -oP '<\K[0-9]+' || true)
if [ -z "$JOB_ID" ]; then
  echo "Failed to get job ID from bsub." >&2
  exit 1
fi
echo "Job ID: $JOB_ID"
ERR_FILE="$LSF_ERR_DIR/benchmark-${JOB_ID}.err"
echo "Waiting for job (polling every 10s, up to 15 min)..."
for i in $(seq 1 90); do
  status=$(bjobs -noheader -o "stat" "$JOB_ID" 2>/dev/null || true)
  [ -z "$status" ] && status="DONE"
  echo "  $i/90: $status"
  if [ "$status" = "DONE" ] || [ "$status" = "EXIT" ]; then
    break
  fi
  sleep 10
done
echo ""
echo "=== stderr ($ERR_FILE) ==="
if [ -f "$ERR_FILE" ]; then
  cat "$ERR_FILE"
else
  echo "(file not found)"
fi
echo ""
echo "=== last 80 lines of stdout ==="
OUT_FILE="$LSF_LOG_DIR/benchmark-${JOB_ID}.out"
if [ -f "$OUT_FILE" ]; then
  tail -80 "$OUT_FILE"
else
  echo "(file not found)"
fi
rm -f "$TEMP_SCRIPT"
