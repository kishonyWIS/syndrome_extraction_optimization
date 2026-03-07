#!/bin/bash
# Submit benchmark sweep: d in {3,5,7,9}, 10M shots, 300 max_errors per config,
# split into multiple jobs per (distance, circuit_type, ...) for faster completion.
# Uses build_circuit_only → sinter collect → sinter_csv_to_benchmark_row (same as d3 test).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLUSTER_DIR="$SCRIPT_DIR"
cd "$REPO_ROOT"

mkdir -p "$CLUSTER_DIR/log" "$CLUSTER_DIR/error" "$CLUSTER_DIR/output" "$CLUSTER_DIR/job_scripts"

# -----------------------------------------------------------------------------
# Sweep (matches user request)
# -----------------------------------------------------------------------------
DISTANCES=(7)
#(3 5 7 9)
ERROR_RATES=(0.001)
NOISE_MODELS=(tqec_uniform_depolarizing)
CIRCUIT_TYPES=(midout optimized_parallel tri_optimal superdense)
# CIRCUIT_TYPES=(optimized_parallel)
# CIRCUIT_TYPES=(superdense)
# CIRCUIT_TYPES=(midout)
DECODERS=(tesseract)

# Total per config (aggregate_results.py merges chunks)
# TOTAL_SHOTS=10000000
TOTAL_MAX_ERRORS=5000
TOTAL_SHOTS=5000000

# Chunking: multiple jobs per config (each runs up to N_SHOTS_PER_CHUNK / MAX_ERRORS_PER_CHUNK)
CHUNKS_PER_CONFIG=100
N_SHOTS_PER_CHUNK=$((TOTAL_SHOTS / CHUNKS_PER_CONFIG))
MAX_ERRORS_PER_CHUNK=$((TOTAL_MAX_ERRORS / CHUNKS_PER_CONFIG))

# optimized_parallel needs schedule CSV
SCHEDULE_CSV="${SCHEDULE_CSV:-$BENCHMARK_DIR/results/zero_collision_schedules.csv}"

NUM_WORKERS=4

# -----------------------------------------------------------------------------
# LSF (email when jobs finish: done, exit, or suspend)
# -----------------------------------------------------------------------------
export LSB_JOB_REPORT_MAIL=N
LSF_QUEUE="${LSF_QUEUE:-berg}"
LSF_NCPUS=4
# 400 was too low: jobs often OOM'd after first sinter flush (~17 shots). Bump so chunks reach 500k.
LSF_MEM=800
LSF_WALLTIME="10:00"
LSF_LOG_DIR="$CLUSTER_DIR/log"
LSF_ERR_DIR="$CLUSTER_DIR/error"
# Whitelist of nodes to use (jobs run ONLY on these via bsub -m). Set LSF_ALLOWED_HOSTS="" to disable.
# If unset, falls back to blacklist (exclude cn3xx). Duplicates are removed.
LSF_ALLOWED_HOSTS="${LSF_ALLOWED_HOSTS:-cn087 cn100 cn104 cn113 cn114 cn118 cn119 cn127 cn131 cn133 cn414 cn418 cn438 cn442 cn444 cn447 cn452 cn469 cn472 cn473 cn479 cn741 cn742 cn754 cn759 cn763 cn785 cn787 cn788}"
LSF_HOST_SELECT=""
LSF_M_HOSTLIST=""   # -m "host1 host2 ..." for strict whitelist (takes precedence)
if [ -n "$LSF_ALLOWED_HOSTS" ]; then
  ALLOWED=$(echo "$LSF_ALLOWED_HOSTS" | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ $//')
  LSF_ALLOWED_LIST="$ALLOWED"
  LSF_M_HOSTLIST="$ALLOWED"
else
  # Fallback: blacklist cn3xx (exclude hosts that cause Illegal instruction)
  if [ -z "${EXCLUDE_HOSTS+x}" ]; then
    EXCLUDE_HOSTS=$(bhosts -w 2>/dev/null | awk 'NR>1 && $1~/^cn3/ {print $1}' | tr '\n' ' ' | sed 's/ $//')
    [ -z "$EXCLUDE_HOSTS" ] && EXCLUDE_HOSTS="cn349 cn358 cn379"
  fi
  if [ -n "$EXCLUDE_HOSTS" ]; then
    LSF_HOST_SELECT="select[$(echo "$EXCLUDE_HOSTS" | sed 's/ \+/ \&\& hname!=/g' | sed 's/^/hname!=/')]"
  fi
fi

JOB_ID_FILE="$CLUSTER_DIR/job_ids_sweep.txt"
touch "$JOB_ID_FILE"

# Unique run ID so re-runs don't overwrite existing chunk files (aggregate_results merges all result_*.csv)
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
echo "Run ID: $RUN_ID (set RUN_ID=... to override)"
if [ -n "${LSF_ALLOWED_LIST:-}" ]; then
  echo "Allowed hosts (whitelist): $LSF_ALLOWED_LIST"
else
  [ -n "$LSF_HOST_SELECT" ] && echo "Host restriction: $LSF_HOST_SELECT"
fi

TOTAL_JOBS=$((${#DISTANCES[@]} * ${#ERROR_RATES[@]} * ${#NOISE_MODELS[@]} * ${#CIRCUIT_TYPES[@]} * ${#DECODERS[@]} * CHUNKS_PER_CONFIG))
echo "Submitting $TOTAL_JOBS jobs (${CHUNKS_PER_CONFIG} chunks per config)"
echo "  Distances: ${DISTANCES[*]}"
echo "  Circuit types: ${CIRCUIT_TYPES[*]}"
echo "  Per chunk: $N_SHOTS_PER_CHUNK shots, $MAX_ERRORS_PER_CHUNK max_errors"
echo "  Total per config: $TOTAL_SHOTS shots, $TOTAL_MAX_ERRORS max_errors"
echo ""

JOB_NUM=0
for noise_model in "${NOISE_MODELS[@]}"; do
  OUT_BASE="$CLUSTER_DIR/output/$noise_model"
  mkdir -p "$OUT_BASE"

  for distance in "${DISTANCES[@]}"; do
    ROUNDS=$distance   # rounds = d
    for error_rate in "${ERROR_RATES[@]}"; do
      for circuit_type in "${CIRCUIT_TYPES[@]}"; do
        for decoder in "${DECODERS[@]}"; do
          chunk=0
          while [ "$chunk" -lt "$CHUNKS_PER_CONFIG" ]; do
            JOB_NUM=$((JOB_NUM + 1))
            OUTPUT_FILE="$OUT_BASE/result_${distance}_${error_rate}_${circuit_type}_${decoder}_chunk${chunk}_${RUN_ID}.csv"
            JOB_NAME="sweep_d${distance}_${circuit_type:0:4}_c${chunk}"

            echo "[$JOB_NUM/$TOTAL_JOBS] d=$distance $circuit_type chunk=$chunk -> $OUTPUT_FILE"

            # Keep line continuation unbroken so --output is always passed
            SCHEDULE_LINE="  \\"
            [ "$circuit_type" = "optimized_parallel" ] && [ -n "${SCHEDULE_CSV:-}" ] && [ -f "$SCHEDULE_CSV" ] && SCHEDULE_LINE="  --schedule-csv $SCHEDULE_CSV \\"

            TEMP_SCRIPT="$CLUSTER_DIR/job_scripts/benchmark_sweep_$$_${JOB_NUM}.sh"
            cat > "$TEMP_SCRIPT" << WRAPPER_EOF
#!/bin/bash
#BSUB -q $LSF_QUEUE
#BSUB -n $LSF_NCPUS
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model!=AMD_EPYC]"
$([ -n "${LSF_M_HOSTLIST:-}" ] && echo "#BSUB -m \"$LSF_M_HOSTLIST\"")
$([ -n "${LSF_HOST_SELECT:-}" ] && echo "#BSUB -R \"$LSF_HOST_SELECT\"")
$([ -n "${LSF_SELECT_EXTRA:-}" ] && echo "#BSUB -R \"$LSF_SELECT_EXTRA\"")
#BSUB -R "rusage[mem=$LSF_MEM]"
#BSUB -M $LSF_MEM
#BSUB -W $LSF_WALLTIME
#BSUB -o $LSF_LOG_DIR/benchmark-%J.out
#BSUB -e $LSF_ERR_DIR/benchmark-%J.err

# Before any Python/module (avoids Illegal instruction on older CPUs, e.g. cn379)
export NPY_DISABLE_CPU_FEATURES="X86_V4,X86_V3"
export OPENBLAS_CPU_FEATURES=""
export MKL_DEBUG_CPU_TYPE=5

set -euo pipefail
cd "$REPO_ROOT"
module load Python/3.11.3-GCCcore-12.3.0
source "$REPO_ROOT/venv/bin/activate"

rm -f "$OUTPUT_FILE"
CIRCUIT_FILE="\${TMPDIR:-/tmp}/circuit_\$\$.stim"
SINTER_CSV="\${TMPDIR:-/tmp}/sinter_\$\$.csv"

# 1) Build circuit (rounds = d)
python3 "$REPO_ROOT/syndrome_extraction_optimization/cluster_benchmark/build_circuit_only.py" \\
  --distance $distance --error-rate $error_rate --noise-model $noise_model \\
  --circuit-type $circuit_type --rounds $ROUNDS \\
$SCHEDULE_LINE
  --output "\$CIRCUIT_FILE"

# 2) Sinter via CLI (log command so we can verify max_shots in job logs)
export PYTHONPATH="$REPO_ROOT/syndrome_extraction_optimization:$REPO_ROOT"
echo "sinter collect --circuits \$CIRCUIT_FILE --max_shots $N_SHOTS_PER_CHUNK --max_errors $MAX_ERRORS_PER_CHUNK --processes $NUM_WORKERS ..." >&2
sinter collect \\
  --circuits "\$CIRCUIT_FILE" \\
  --decoders $decoder \\
  --custom_decoders_module_function "benchmark_circuits:get_custom_decoders" \\
  --max_shots $N_SHOTS_PER_CHUNK \\
  --max_errors $MAX_ERRORS_PER_CHUNK \\
  --processes $NUM_WORKERS \\
  --save_resume_filepath "\$SINTER_CSV"

# 3) Convert to one-row benchmark CSV
python3 "$REPO_ROOT/syndrome_extraction_optimization/cluster_benchmark/sinter_csv_to_benchmark_row.py" \\
  "\$SINTER_CSV" -o "$OUTPUT_FILE" \\
  --distance $distance --rounds $ROUNDS --p-cnot $error_rate \\
  --circuit-type $circuit_type --decoder $decoder --noise-model $noise_model \\
  --n-shots $N_SHOTS_PER_CHUNK

rm -f "\$CIRCUIT_FILE" "\$SINTER_CSV"
rm -f "\$0"
WRAPPER_EOF

            chmod +x "$TEMP_SCRIPT"
            # Pass -m on command line so scheduler strictly restricts to whitelist (some clusters ignore #BSUB -m in script)
            BSUB_OPTS=(-J "$JOB_NAME")
            [ -n "${LSF_M_HOSTLIST:-}" ] && BSUB_OPTS+=(-m "$LSF_M_HOSTLIST")
            JOB_ID=$(bsub "${BSUB_OPTS[@]}" "$TEMP_SCRIPT" 2>&1 | grep -oP '<\K[0-9]+' || true)

            if [ -n "${JOB_ID:-}" ]; then
              echo "$JOB_ID $noise_model $distance $error_rate $circuit_type $decoder $chunk" >> "$JOB_ID_FILE"
            fi
            chunk=$((chunk + 1))
            sleep 0.1
          done
        done
      done
    done
  done
done

echo ""
echo "Submitted $TOTAL_JOBS jobs. Job IDs: $JOB_ID_FILE"
echo "Monitor: bjobs"
echo "After completion: python3 syndrome_extraction_optimization/cluster_benchmark/aggregate_results.py --output-dir $CLUSTER_DIR/output"
