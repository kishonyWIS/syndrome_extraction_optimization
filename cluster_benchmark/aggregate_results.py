#!/usr/bin/env python3
"""
Aggregate chunk/result CSVs from cluster benchmark jobs.

Finds all result_*.csv under an output directory, groups by
(distance, rounds, p_cnot, circuit_type, decoder, noise_model),
sums shots_used and errors, recomputes error_rate and 95% CI,
and writes a single combined CSV. Supports incremental runs: if
aggregated_results.csv already exists, new chunk data is merged in
(summed by config) and chunk files are deleted after incorporation.
Plotting uses only the CSV (no pickle).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from scipy import stats


GROUP_COLS = ["distance", "rounds", "p_cnot", "circuit_type", "decoder", "noise_model"]
OUT_COLS = [
    "distance", "rounds", "p_cnot", "circuit_type", "decoder", "noise_model",
    "circuit_distance", "n_shots", "shots_used", "errors",
    "error_rate", "ci_low", "ci_high", "decode_time", "n_chunks",
]


def aggregate_group(g):
    total_shots = g["shots_used"].sum()
    total_errors = g["errors"].sum()
    if total_shots == 0:
        error_rate = None
        ci_low, ci_high = None, None
    else:
        error_rate = total_errors / total_shots
        if total_errors > 0:
            ci = stats.binomtest(int(total_errors), int(total_shots)).proportion_ci(
                confidence_level=0.95
            )
            ci_low, ci_high = ci.low, ci.high
        else:
            ci_low, ci_high = 0.0, 1.0 / total_shots

    return pd.Series({
        "circuit_distance": g["circuit_distance"].iloc[0] if "circuit_distance" in g.columns else None,
        "n_shots": g["n_shots"].iloc[0] if "n_shots" in g.columns else None,
        "shots_used": total_shots,
        "errors": total_errors,
        "error_rate": error_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "decode_time": g["decode_time"].sum() if "decode_time" in g.columns else None,
        "n_chunks": len(g),
    })


def aggregate_raw_dfs(dfs, group_cols):
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    if "noise_model" not in combined.columns:
        combined["noise_model"] = "depolarize2_after_cnot"
    for c in group_cols:
        if c not in combined.columns:
            raise ValueError(f"missing column {c}")
    agg = combined.groupby(group_cols, as_index=False).apply(aggregate_group)
    return agg.reset_index(drop=True)


def merge_aggregates(existing_agg, new_agg, group_cols):
    """Merge two aggregated DataFrames by summing shots_used, errors, decode_time, n_chunks; recompute CI."""
    if existing_agg is None or existing_agg.empty:
        return new_agg
    if new_agg is None or new_agg.empty:
        return existing_agg

    merged = existing_agg.merge(
        new_agg,
        on=group_cols,
        how="outer",
        suffixes=("_old", "_new"),
    )
    # Sum numeric columns; prefer _new then _old for scalars
    for col in ["shots_used", "errors", "decode_time", "n_chunks"]:
        if f"{col}_old" in merged.columns and f"{col}_new" in merged.columns:
            merged[col] = merged[f"{col}_old"].fillna(0) + merged[f"{col}_new"].fillna(0)
        elif f"{col}_old" in merged.columns:
            merged[col] = merged[f"{col}_old"]
        else:
            merged[col] = merged[f"{col}_new"]

    # Recompute error_rate and CI from combined shots/errors
    def row_ci(row):
        s, e = int(row["shots_used"]), int(row["errors"])
        if s == 0:
            return None, None, None
        rate = e / s
        if e > 0:
            ci = stats.binomtest(e, s).proportion_ci(confidence_level=0.95)
            return rate, ci.low, ci.high
        return rate, 0.0, 1.0 / s

    out = merged[group_cols + ["shots_used", "errors", "decode_time", "n_chunks"]].copy()
    out[["error_rate", "ci_low", "ci_high"]] = out.apply(
        lambda r: row_ci(r), axis=1, result_type="expand"
    )
    # Preserve circuit_distance, n_shots from either side (first non-null)
    for scalar in ["circuit_distance", "n_shots"]:
        old_c = f"{scalar}_old" if f"{scalar}_old" in merged.columns else scalar
        new_c = f"{scalar}_new" if f"{scalar}_new" in merged.columns else scalar
        if old_c in merged.columns and new_c in merged.columns:
            out[scalar] = merged[new_c].fillna(merged[old_c])
        elif old_c in merged.columns:
            out[scalar] = merged[old_c]
        else:
            out[scalar] = merged[new_c]

    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = None
    return out[[c for c in OUT_COLS if c in out.columns]]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cluster benchmark result CSVs into one combined file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory containing result CSVs (e.g. cluster_benchmark/output). "
             "Defaults to syndrome_extraction_optimization/cluster_benchmark/output.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/result_*.csv",
        help="Glob pattern relative to output-dir (default: **/result_*.csv)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output combined CSV path. Default: <output_dir>/aggregated_results.csv",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not delete raw chunk CSVs after incorporating into aggregate.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir / "output"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    if not output_dir.is_dir():
        print(f"Error: output-dir not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    out_csv = args.out_csv or str(output_dir / "aggregated_results.csv")
    agg_path = Path(out_csv)

    # Load existing aggregate if present
    existing_agg = None
    if agg_path.is_file():
        try:
            existing_agg = pd.read_csv(agg_path)
            if existing_agg.empty:
                existing_agg = None
        except Exception as e:
            print(f"Warning: could not load existing aggregate {agg_path}: {e}", file=sys.stderr)

    pattern = args.pattern
    csv_files = sorted(output_dir.glob(pattern))
    # Exclude the aggregate file itself if it lives under output_dir
    csv_files = [p for p in csv_files if p.resolve() != agg_path.resolve()]

    if not csv_files and existing_agg is None:
        print(f"No CSVs matching {pattern} under {output_dir} and no existing aggregate.", file=sys.stderr)
        sys.exit(1)

    dfs = []
    read_ok = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            if df.empty:
                read_ok.append(p)
                continue
            dfs.append(df)
            read_ok.append(p)
        except Exception as e:
            print(f"Warning: skip {p}: {e}", file=sys.stderr)

    new_agg = None
    if dfs:
        for c in GROUP_COLS:
            if c not in dfs[0].columns:
                print(f"Error: missing column {c} in chunk CSVs", file=sys.stderr)
                sys.exit(1)
        new_agg = aggregate_raw_dfs(dfs, GROUP_COLS)
    aggregated = merge_aggregates(existing_agg, new_agg, GROUP_COLS)

    if aggregated is None or aggregated.empty:
        print("No data to write.", file=sys.stderr)
        sys.exit(1)

    out_csv_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)
    aggregated.to_csv(out_csv, index=False)

    deleted = 0
    if not args.no_delete and read_ok:
        for p in read_ok:
            try:
                p.unlink()
                deleted += 1
            except OSError as e:
                print(f"Warning: could not remove {p}: {e}", file=sys.stderr)

    total_rows = len(aggregated)
    chunk_count = len(read_ok)
    print(f"Aggregated -> {total_rows} configs (merged {chunk_count} chunk files into {out_csv})")
    if deleted:
        print(f"Removed {deleted} raw chunk file(s).")


if __name__ == "__main__":
    main()
