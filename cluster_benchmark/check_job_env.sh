#!/bin/bash
# Run this ON THE CLUSTER via: bsub -q berg -W 5 -o cluster_benchmark/log/check_env.out -e cluster_benchmark/log/check_env.err bash syndrome_extraction_optimization/cluster_benchmark/check_job_env.sh
# Then: cat syndrome_extraction_optimization/cluster_benchmark/error/check_env.err  (or log/check_env.err depending on your bsub -e path)
# If you see "ImportError.*libbz2" then the job-script LD_LIBRARY_PATH fix is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

module load Python/3.11.3-GCCcore-12.3.0
export LD_LIBRARY_PATH="${EBROOTPYTHON:-/apps/easybd/easybuild/amd/software/Python/3.11.3-GCCcore-12.3.0}/lib:${LD_LIBRARY_PATH:-}"
source "$REPO_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/syndrome_extraction_optimization:$REPO_ROOT"
export NPY_DISABLE_CPU_FEATURES="X86_V4,X86_V3"

echo "Python: $(which python3)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Testing import: benchmark_circuits.get_custom_decoders ..."

python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT/syndrome_extraction_optimization')
sys.path.insert(0, '$REPO_ROOT')
try:
    from benchmark_circuits import get_custom_decoders
    get_custom_decoders()
    print('OK: benchmark_circuits imported and get_custom_decoders() ran')
except Exception as e:
    print('FAIL:', type(e).__name__, str(e), file=sys.stderr)
    raise SystemExit(1)
"

echo "Done."
