#!/usr/bin/env python3
"""
Single benchmark task runner for cluster execution.

Runs one (distance, error_rate, noise_model, circuit_type, decoder) configuration
with given n_shots and max_errors, and writes one row to a CSV file.
Designed to be called from LSF job scripts; supports chunked runs via --chunk-id.
"""

import argparse
import csv
import os
import sys

# Ensure repo root and syndrome_extraction_optimization are on path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)  # syndrome_extraction_optimization
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
for path in (REPO_ROOT, BENCHMARK_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# gidney_circuits path is relative to cwd; submit script should cd to repo root
os.chdir(REPO_ROOT)

from benchmark_circuits import (
    apply_noise,
    build_parallel_circuit,
    build_tri_optimal_circuit,
    compute_circuit_distance_stim,
    generate_gidney_circuit,
    load_zero_collision_schedule,
    run_sinter_single_task,
    TRI_OPTIMAL_SCHEDULE_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single benchmark task (one circuit + decoder + p_cnot + noise_model)."
    )
    parser.add_argument("--distance", type=int, required=True, help="Code distance")
    parser.add_argument("--error-rate", type=float, required=True, dest="error_rate", help="p_cnot / physical error rate")
    parser.add_argument("--noise-model", type=str, required=True, help="Noise model name")
    parser.add_argument("--circuit-type", type=str, required=True, help="optimized_parallel | tri_optimal | midout | superdense")
    parser.add_argument("--decoder", type=str, required=True, help="Decoder name (e.g. tesseract)")
    parser.add_argument("--n-shots", type=int, required=True, help="Max shots for this task/chunk")
    parser.add_argument("--max-errors", type=int, required=True, help="Max errors for early stopping")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path (one row)")
    parser.add_argument("--chunk-id", type=int, default=None, help="Optional chunk index (for filename only)")
    parser.add_argument("--rounds", type=int, default=None, help="Syndrome rounds; default = distance")
    parser.add_argument("--schedule-csv", type=str, default=None, help="Path to zero_collision_schedules.csv for optimized_parallel")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for sinter")
    parser.add_argument("--skip-circuit-distance", action="store_true", help="Skip circuit-distance computation (faster for quick tests)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    rounds = args.rounds if args.rounds is not None else args.distance
    circuit_type = args.circuit_type
    noise_model = args.noise_model
    p_cnot = args.error_rate
    d = args.distance
    decoder = args.decoder

    # Load schedule for optimized_parallel if needed
    optimized_schedule = None
    if circuit_type == "optimized_parallel":
        schedule_csv = args.schedule_csv or os.path.join(BENCHMARK_DIR, "results", "zero_collision_schedules.csv")
        if not os.path.isfile(schedule_csv):
            # Try repo-root relative
            schedule_csv = args.schedule_csv or "syndrome_extraction_optimization/results/zero_collision_schedules.csv"
        if not os.path.isfile(schedule_csv) and args.schedule_csv:
            schedule_csv = args.schedule_csv
        if not os.path.isfile(schedule_csv):
            raise FileNotFoundError(
                f"optimized_parallel requires zero_collision_schedules.csv; not found at {schedule_csv}. Use --schedule-csv."
            )
        optimized_schedule = load_zero_collision_schedule(schedule_csv, index=0)

    # Build noiseless circuit
    if circuit_type == "optimized_parallel":
        if optimized_schedule is None:
            raise ValueError("optimized_parallel requires --schedule-csv")
        circuit = build_parallel_circuit(d, rounds, p_cnot, optimized_schedule)
    elif circuit_type == "tri_optimal":
        circuit = build_tri_optimal_circuit(d, rounds, p_cnot)
    elif circuit_type in ("midout", "superdense"):
        circuit = generate_gidney_circuit(circuit_type, d, rounds, p_cnot)
    else:
        raise ValueError(f"Unknown circuit_type: {circuit_type!r}")

    circuit = apply_noise(circuit, p_cnot, noise_model)

    # Circuit-level distance (same for all p_cnot for this circuit_type, d)
    if args.skip_circuit_distance:
        circuit_distance = None
    else:
        try:
            circuit_distance = compute_circuit_distance_stim(circuit)
        except Exception:
            circuit_distance = None

    # Run sinter
    metadata = {
        "distance": d,
        "rounds": rounds,
        "p_cnot": p_cnot,
        "circuit_type": circuit_type,
        "decoder": decoder,
        "noise_model": noise_model,
    }
    result = run_sinter_single_task(
        circuit=circuit,
        decoder=decoder,
        metadata=metadata,
        max_shots=args.n_shots,
        max_errors=args.max_errors,
        num_workers=args.num_workers,
    )

    if not result:
        print("No result from sinter", file=sys.stderr)
        sys.exit(1)

    row = {
        "distance": d,
        "rounds": rounds,
        "p_cnot": p_cnot,
        "circuit_type": circuit_type,
        "decoder": decoder,
        "noise_model": noise_model,
        "circuit_distance": circuit_distance,
        "n_shots": args.n_shots,
        "shots_used": result["shots_used"],
        "errors": result["errors"],
        "error_rate": result["error_rate"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "decode_time": result["decode_time"],
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "distance", "rounds", "p_cnot", "circuit_type", "decoder", "noise_model",
        "circuit_distance", "n_shots", "shots_used", "errors",
        "error_rate", "ci_low", "ci_high", "decode_time",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"Wrote {args.output}: {result['errors']}/{result['shots_used']} errors, rate={result['error_rate']}")


if __name__ == "__main__":
    main()
