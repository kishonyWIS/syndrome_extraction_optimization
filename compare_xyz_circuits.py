#!/usr/bin/env python3
"""
Compare optimized_parallel_XYZ (build_xyz_circuit) vs stim_memory_xyz circuits.
Prints both so detector structure can be compared.
"""
import sys
sys.path.insert(0, ".")

from benchmark_circuits import (
    build_stim_memory_xyz_circuit,
    build_xyz_circuit,
    TRI_OPTIMAL_SCHEDULE_CONFIG,
    apply_noise,
)

def main():
    d = 5
    rounds = 5
    p = 0.0
    p_noise = 0.001  # si1000 noise strength

    print("=" * 70)
    print("stim_memory_xyz (Stim built-in) + si1000 noise")
    print("=" * 70)
    c_stim = build_stim_memory_xyz_circuit(d, rounds, p)
    # c_stim = apply_noise(c_stim, p_noise, "si1000")
    print(c_stim)
    print(f"\n# Detectors only (stim_memory_xyz):")

    # create detector error model
    dem_stim = c_stim.detector_error_model()
    print(dem_stim[0:10])
    print("\n")
    print("=" * 70)
    print("optimized_parallel_XYZ (build_xyz_circuit, TRI_OPTIMAL schedule) + si1000 noise")
    print("=" * 70)
    c_xyz = build_xyz_circuit(d, rounds, p, TRI_OPTIMAL_SCHEDULE_CONFIG)
    # c_xyz = apply_noise(c_xyz, p_noise, "si1000")
    print(c_xyz)
    print(f"\n# Detectors only (optimized_parallel_XYZ):")

    dem_xyz = c_xyz.detector_error_model()
    print(dem_xyz[0:10])

    print("\n# Summary:")
    print(f"  stim_memory_xyz: {c_stim.num_qubits} qubits, {c_stim.num_detectors} detectors")
    print(f"  optimized_parallel_XYZ: {c_xyz.num_qubits} qubits, {c_xyz.num_detectors} detectors")


if __name__ == "__main__":
    main()
