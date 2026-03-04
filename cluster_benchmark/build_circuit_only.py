#!/usr/bin/env python3
"""
Build a single benchmark circuit and write it to a .stim file.
No sinter, no multiprocessing. Used so the job can then run sinter via CLI.
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
for path in (REPO_ROOT, BENCHMARK_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO_ROOT)

from benchmark_circuits import (
    apply_noise,
    build_parallel_circuit,
    build_tri_optimal_circuit,
    generate_gidney_circuit,
    load_zero_collision_schedule,
)


def main():
    p = argparse.ArgumentParser(description="Build circuit and write to .stim file.")
    p.add_argument("--distance", type=int, required=True)
    p.add_argument("--error-rate", type=float, required=True, dest="error_rate")
    p.add_argument("--noise-model", type=str, required=True)
    p.add_argument("--circuit-type", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="Output .stim file path")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--schedule-csv", type=str, default=None)
    args = p.parse_args()

    rounds = args.rounds if args.rounds is not None else args.distance
    circuit_type = args.circuit_type
    p_cnot = args.error_rate
    d = args.distance
    noise_model = args.noise_model

    optimized_schedule = None
    if circuit_type == "optimized_parallel":
        schedule_csv = args.schedule_csv or os.path.join(BENCHMARK_DIR, "results", "zero_collision_schedules.csv")
        if not os.path.isfile(schedule_csv):
            schedule_csv = args.schedule_csv
        if not args.schedule_csv or not os.path.isfile(schedule_csv):
            raise FileNotFoundError("optimized_parallel requires --schedule-csv pointing to zero_collision_schedules.csv")
        optimized_schedule = load_zero_collision_schedule(schedule_csv, index=0)

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
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    circuit.to_file(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
