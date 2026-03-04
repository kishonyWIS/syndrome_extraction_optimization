#!/usr/bin/env python3
"""
Plot aggregated cluster benchmark results using sinter.plot_error_rate.

Loads aggregated_results.csv, converts to sinter.TaskStats, and produces
the qubit-scaling plot (error rate per round vs qubits) with proper
uncertainty bands and styling.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_qubit_count(d: int, circuit_type: str) -> int:
    """Total qubits for triangular color code: data + aux_per_plaquette * num_plaquettes."""
    data_qubits = (3 * d * d + 1) // 4
    num_plaquettes = 3 * (d * d - 1) // 8
    if circuit_type == "superdense":
        aux_per_plaquette = 2
    elif circuit_type in ("tri_optimal", "optimized_parallel"):
        aux_per_plaquette = 1
    elif circuit_type == "midout":
        aux_per_plaquette = 0
    else:
        aux_per_plaquette = 1
    return data_qubits + aux_per_plaquette * num_plaquettes


def df_to_task_stats(df: pd.DataFrame) -> list:
    """Convert aggregated DataFrame rows to sinter.TaskStats for plotting."""
    stats = []
    for _, row in df.iterrows():
        d, r = int(row["distance"]), int(row["rounds"])
        ct = row["circuit_type"]
        num_qubits = compute_qubit_count(d, ct)
        meta = {
            "d": d,
            "r": r,
            "p": float(row["p_cnot"]),
            "circuit_type": ct,
            "num_qubits": num_qubits,
        }
        stat = sinter.TaskStats(
            strong_id=f"{d}_{ct}_{row['p_cnot']}_{row['decoder']}",
            decoder=row["decoder"],
            json_metadata=meta,
            shots=int(row["shots_used"]),
            errors=int(row["errors"]),
            discards=0,
            seconds=float(row["decode_time"]) if pd.notna(row.get("decode_time")) else 0,
        )
        stats.append(stat)
    return stats


# Circuit type styling (match benchmark_circuits.plot_error_vs_qubits)
CIRCUIT_COLORS = {
    "optimized_parallel": "#2ecc71",
    "tri_optimal": "#3498db",
    "midout": "#e74c3c",
    "superdense": "#9b59b6",
}
CIRCUIT_MARKERS = {
    "optimized_parallel": "o",
    "tri_optimal": "s",
    "midout": "^",
    "superdense": "D",
}
CIRCUIT_LABELS = {
    "midout": "0 aux. per plaq. (midout)",
    "superdense": "2 aux. per plaq. (superdense)",
    "tri_optimal": "1 aux. per plaq. uniform",
    "optimized_parallel": "1 aux. per plaq. non-uniform",
}
LEGEND_ORDER = ["midout", "superdense", "tri_optimal", "optimized_parallel"]


def group_func(stat: sinter.TaskStats) -> dict:
    """Return curve styling dict for sinter.plot_error_rate (label, color, marker, sort)."""
    ct = stat.json_metadata["circuit_type"]
    sort_idx = LEGEND_ORDER.index(ct) if ct in LEGEND_ORDER else 99
    return {
        "label": CIRCUIT_LABELS.get(ct, ct),
        "color": CIRCUIT_COLORS.get(ct, "#333333"),
        "marker": CIRCUIT_MARKERS.get(ct, "o"),
        "sort": sort_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated benchmark results (error vs qubits).")
    parser.add_argument(
        "csv",
        nargs="?",
        default=os.path.join(SCRIPT_DIR, "output", "aggregated_results.csv"),
        help="Path to aggregated_results.csv",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=None,
        help="Decoder to plot (default: first in data)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the plot (default: show only)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if df.empty:
        print("Error: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    if args.decoder is not None:
        df = df[df["decoder"] == args.decoder].copy()
    else:
        args.decoder = df["decoder"].iloc[0] if "decoder" in df.columns else "unknown"
    if df.empty:
        print(f"No data for decoder: {args.decoder}", file=sys.stderr)
        sys.exit(1)

    # Add num_qubits for metadata (we'll put it in TaskStats.json_metadata)
    stats = df_to_task_stats(df)
    error_rates = sorted(df["p_cnot"].unique())
    n_plots = len(error_rates)

    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8), squeeze=False)
    axes = axes.flatten()

    for ax_idx, p_cnot in enumerate(error_rates):
        ax = axes[ax_idx]
        subset = [s for s in stats if s.json_metadata["p"] == p_cnot]
        if not subset:
            continue

        sinter.plot_error_rate(
            ax=ax,
            stats=subset,
            x_func=lambda s: s.json_metadata["num_qubits"],
            failure_units_per_shot_func=lambda s: s.json_metadata["r"],
            group_func=group_func,
            filter_func=lambda _: True,
            highlight_max_likelihood_factor=1000.0,
            point_label_func=lambda s: f"d={s.json_metadata['d']}",
        )

        # sqrt scale on x-axis (actual qubit counts, spaced by sqrt)
        ax.set_xscale("function", functions=(np.sqrt, np.square))
        ax.set_yscale("log")

        all_qubits = sorted({s.json_metadata["num_qubits"] for s in subset})
        ax.set_xticks(all_qubits)
        ax.set_xticklabels([str(int(q)) for q in all_qubits])
        ax.set_xlabel("Total Qubits (sqrt scale)", fontsize=22)
        ax.set_ylabel("Logical Error Rate (per round)", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.tick_params(axis="both", which="minor", labelsize=22)
        ax.grid(True, alpha=0.3)

        # Limits
        x_vals = [s.json_metadata["num_qubits"] for s in subset]
        x_min, x_max = min(x_vals) * 0.8, max(x_vals) * 1.1
        ax.set_xlim(x_min, x_max)
        # Extend y-axis below lowest data: use per-round rate lower bound from data
        y_vals_approx = [
            s.errors / (s.shots * s.json_metadata["r"])
            for s in subset
            if s.shots > 0 and s.json_metadata["r"] > 0
        ]
        y_min_data = min(y_vals_approx) if y_vals_approx else 1e-5
        y_min = max(1e-7, y_min_data)
        y_max = max(y_vals_approx) if y_vals_approx else 1e-2
        ax.set_ylim(y_min*0.2, y_max*2)

        # Ensure legend is visible and bold last entry (optimized_parallel)
        leg = ax.get_legend()
        if leg is None:
            leg = ax.legend(loc="best", fontsize=22)
        if leg:
            if leg.get_texts():
                leg.get_texts()[-1].set_fontweight("bold")
            leg.get_frame().set_alpha(0.9)

    fig.set_dpi(150)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.output}")

    plt.show()

    # Print qubit counts for reference
    print("\nQubit counts by circuit type and distance:")
    for circuit_type in df["circuit_type"].unique():
        print(f"  {circuit_type}:")
        for d in sorted(df["distance"].unique()):
            qubits = compute_qubit_count(int(d), circuit_type)
            print(f"    d={d}: {qubits} qubits (sqrt={np.sqrt(qubits):.2f})")


if __name__ == "__main__":
    main()
