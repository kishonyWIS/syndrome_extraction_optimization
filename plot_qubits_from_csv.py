#!/usr/bin/env python3
"""
Generate logical error rate vs number of qubits from a benchmark CSV.
Uses the same logic as benchmark_circuits.plot_error_vs_qubits.
"""
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_qubit_count(d: int, circuit_type: str) -> int:
    """Total qubits for triangular color code (same as benchmark_circuits)."""
    data_qubits = (3 * d * d + 1) // 4
    num_plaquettes = 3 * (d * d - 1) // 8
    if circuit_type in ['superdense', 'superdense_modified']:
        aux_per_plaquette = 2
    elif circuit_type in ['tri_optimal', 'tri_optimal_XYZ', 'optimized_parallel', 'optimized_parallel_XYZ', 'stim_memory_xyz']:
        aux_per_plaquette = 1
    elif circuit_type == 'midout':
        aux_per_plaquette = 0
    else:
        aux_per_plaquette = 1
    return data_qubits + aux_per_plaquette * num_plaquettes


def plot_error_vs_qubits(df: pd.DataFrame, decoder: str = None, save_path: str = None):
    """Logical error rate per round vs number of qubits (sqrt scale on x-axis)."""
    if decoder is not None:
        df_plot = df[df['decoder'] == decoder].copy()
    else:
        df_plot = df.copy()
        decoder = df_plot['decoder'].iloc[0] if not df_plot.empty else 'unknown'

    if df_plot.empty:
        print(f"No data for decoder: {decoder}", file=sys.stderr)
        return

    df_plot['num_qubits'] = df_plot.apply(
        lambda row: compute_qubit_count(row['distance'], row['circuit_type']), axis=1
    )
    df_plot['error_rate_per_round'] = df_plot['error_rate'] / df_plot['rounds']
    df_plot['ci_low_per_round'] = df_plot['ci_low'] / df_plot['rounds']
    df_plot['ci_high_per_round'] = df_plot['ci_high'] / df_plot['rounds']

    error_rates = sorted(df_plot['p_cnot'].unique())
    n_plots = len(error_rates)
    if n_plots == 0:
        print("No error rates found", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8), squeeze=False)
    axes = axes.flatten()

    circuit_colors = {
        'optimized_parallel': '#2ecc71',
        'tri_optimal': '#3498db',
        'midout': '#e74c3c',
        'superdense': '#9b59b6',
    }
    circuit_markers = {
        'optimized_parallel': 'o',
        'tri_optimal': 's',
        'midout': '^',
        'superdense': 'D',
    }
    circuit_display_labels = {
        'midout': '0 aux. per plaq. (midout)',
        'superdense': '2 aux. per plaq. (superdense)',
        'tri_optimal': '1 aux. per plaq. uniform',
        'optimized_parallel': '1 aux. per plaq. non-uniform',
    }
    legend_order = ['midout', 'superdense', 'tri_optimal', 'optimized_parallel']

    for ax_idx, p_cnot in enumerate(error_rates):
        ax = axes[ax_idx]
        df_p = df_plot[df_plot['p_cnot'] == p_cnot]
        types_in_data = [ct for ct in legend_order if ct in df_p['circuit_type'].values]
        for circuit_type in types_in_data:
            df_ct = df_p[df_p['circuit_type'] == circuit_type].sort_values('num_qubits')
            color = circuit_colors.get(circuit_type, '#333333')
            marker = circuit_markers.get(circuit_type, 'o')
            label = circuit_display_labels.get(circuit_type, circuit_type)
            yerr_low = df_ct['error_rate_per_round'] - df_ct['ci_low_per_round'].fillna(0)
            yerr_high = df_ct['ci_high_per_round'].fillna(df_ct['error_rate_per_round']) - df_ct['error_rate_per_round']
            ax.errorbar(
                df_ct['num_qubits'],
                df_ct['error_rate_per_round'],
                yerr=[yerr_low.clip(lower=0), yerr_high.clip(lower=0)],
                marker=marker, label=label, color=color, capsize=4, markersize=14, linewidth=4,
            )
            for _, row in df_ct.iterrows():
                ax.annotate(
                    f'd={int(row["distance"])}',
                    (row['num_qubits'], row['error_rate_per_round']),
                    textcoords="offset points", xytext=(5, 5), fontsize=22, alpha=0.7,
                )
        # sqrt scale on x (matplotlib 3.7+); fallback to linear
        try:
            ax.set_xscale('function', functions=(np.sqrt, np.square))
        except (TypeError, ValueError):
            pass  # linear scale
        ax.set_yscale('log')
        all_qubits = sorted(df_p['num_qubits'].unique())
        ax.set_xticks(all_qubits)
        ax.set_xticklabels([str(int(q)) for q in all_qubits])
        ax.set_xlabel('Total Qubits (sqrt scale)', fontsize=22)
        ax.set_ylabel('Logical Error Rate (per round)', fontsize=22)
        ax.legend(fontsize=22, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=22)
        leg = ax.get_legend()
        if leg and leg.get_texts():
            leg.get_texts()[-1].set_fontweight('bold')
        ax.grid(True, alpha=0.3)
        x_min, x_max = df_p['num_qubits'].min() * 0.8, df_p['num_qubits'].max() * 1.1
        ax.set_xlim(x_min, x_max)
        y_min = df_p['error_rate_per_round'].min() * 0.8
        ax.set_ylim(y_min, 1.2e-2)

    fig.set_dpi(150)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


CSV_PATH = 'results/benchmark_results_0.005_depolarize2_after_cnot_fixed_double_midout_rounds_short_beam_tesseract.csv'
OUT_PATH = 'results/qubits_scaling_0.005_short_beam_tesseract.pdf'


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    plot_error_vs_qubits(df, decoder='tesseract', save_path=OUT_PATH)
    if os.path.isfile(OUT_PATH):
        print(f"Saved: {OUT_PATH}")
    else:
        print("Warning: output file was not created", file=sys.stderr)


if __name__ == '__main__':
    main()
