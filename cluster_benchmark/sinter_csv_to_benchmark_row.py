#!/usr/bin/env python3
"""
Read sinter collect output CSV (one or more rows, one per batch) and write a single row
in our benchmark CSV format. Sums shots/errors/seconds across all rows (sinter appends
one row per flush). Computes error_rate and 95% CI from total shots/errors.
"""
import argparse
import csv
import json
import os
import sys
from scipy import stats


def main():
    p = argparse.ArgumentParser(description="Convert sinter CSV row to benchmark CSV row.")
    p.add_argument("sinter_csv", help="Path to CSV produced by sinter collect (header + one row per batch)")
    p.add_argument("--output", "-o", required=True, help="Output CSV path (one row, our format)")
    p.add_argument("--distance", type=int, required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--p-cnot", type=float, required=True, dest="p_cnot")
    p.add_argument("--circuit-type", type=str, required=True)
    p.add_argument("--decoder", type=str, required=True)
    p.add_argument("--noise-model", type=str, required=True)
    p.add_argument("--n-shots", type=int, required=True)
    p.add_argument("--circuit-distance", type=int, default=None, help="Optional; omit to leave blank")
    args = p.parse_args()

    with open(args.sinter_csv) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = list(reader)
    if not rows:
        print("No data rows in sinter CSV", file=sys.stderr)
        sys.exit(1)
    # Sinter appends one CSV row per batch; sum all rows for this run
    shots = 0
    errors = 0
    seconds = 0.0
    for raw in rows:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
        shots += int(row["shots"])
        errors += int(row["errors"])
        seconds += float(row["seconds"])

    if shots > 0:
        error_rate = errors / shots
        if errors > 0:
            ci = stats.binomtest(errors, shots).proportion_ci(confidence_level=0.95)
            ci_low, ci_high = ci.low, ci.high
        else:
            ci_low, ci_high = 0.0, 1.0 / shots
    else:
        error_rate = None
        ci_low, ci_high = None, None

    out_row = {
        "distance": args.distance,
        "rounds": args.rounds,
        "p_cnot": args.p_cnot,
        "circuit_type": args.circuit_type,
        "decoder": args.decoder,
        "noise_model": args.noise_model,
        "circuit_distance": args.circuit_distance,
        "n_shots": args.n_shots,
        "shots_used": shots,
        "errors": errors,
        "error_rate": error_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "decode_time": seconds,
    }
    fieldnames = list(out_row.keys())
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(out_row)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
