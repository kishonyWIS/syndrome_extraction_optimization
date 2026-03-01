"""
Benchmark comparison of color code circuits.

Compares logical error rates across 4 circuit constructions:
1. Optimized schedule (from exhaustive search)
2. tri_optimal schedule (built-in to color_code_stim)
3. Gidney's middle-out circuit
4. Gidney's superdense circuit

All circuits are normalized to use DEPOLARIZE2 after each CNOT only.
"""

import os
import numpy as np
import pandas as pd
import stim
import sinter
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from color_code_stim import ColorCode
from compare_schedules import schedule_to_cnot_dict_by_color_and_type
from tesseract_decoder import tesseract, utils as tesseract_utils
from mwpf import SinterMWPFDecoder


# =============================================================================
# Custom Sinter Decoders (file-based interface to avoid numpy 2.x issues)
# =============================================================================

class TesseractSinterDecoder(sinter.Decoder):
    """Sinter decoder wrapper for Tesseract hypergraph decoder using file-based interface."""
    
    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: str,
        dets_b8_in_path: str,
        obs_predictions_b8_out_path: str,
        tmp_dir: str,
    ) -> None:
        """Decode using file-based interface."""
        # Load DEM
        dem = stim.DetectorErrorModel.from_file(dem_path)
        
        # Create tesseract decoder with short-beam config
        tesseract_config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=200_000,
            det_beam=15,
            beam_climbing=True,
            det_orders=tesseract_utils.build_det_orders(
                dem=dem,
                num_det_orders=16,
                method=tesseract_utils.DetOrder.DetIndex,
            ),
            no_revisit_dets=True,
        )
        decoder = tesseract.TesseractDecoder(tesseract_config)
        
        # Load detector data
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format='b8',
            num_detectors=num_dets,
            num_observables=0,
        )
        
        # Decode batch
        predictions = decoder.decode_batch(dets)
        
        # Write predictions
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format='b8',
            num_observables=num_obs,
        )


# Custom decoder registry for sinter
# Tesseract: hypergraph decoder optimized for color codes
# Hyperion (MWPF): Minimum-Weight Parity Factor decoder for general qLDPC codes
CUSTOM_SINTER_DECODERS = {
    'tesseract': TesseractSinterDecoder(),
    'hyperion': SinterMWPFDecoder(cluster_node_limit=200),
}


# =============================================================================
# Circuit Loading and Normalization
# =============================================================================

def add_depolarize2_after_cnot(circuit: stim.Circuit, p_cnot: float) -> stim.Circuit:
    """
    Add DEPOLARIZE2 error after each CNOT (CX) gate.
    
    Args:
        circuit: Noiseless stim circuit
        p_cnot: Depolarization probability
        
    Returns:
        Circuit with DEPOLARIZE2 after each CX
    """
    new_circuit = stim.Circuit()
    
    for instruction in circuit:
        if instruction.name == "REPEAT":
            # Handle REPEAT blocks recursively
            inner_circuit = instruction.body_copy()
            noisy_inner = add_depolarize2_after_cnot(inner_circuit, p_cnot)
            # Create REPEAT block properly
            repeat_block = stim.Circuit()
            repeat_block += noisy_inner
            new_circuit += repeat_block * instruction.repeat_count
        else:
            new_circuit.append(instruction)
            
            # Add DEPOLARIZE2 after CX gates (only for real qubit targets, not measurement records)
            if instruction.name == "CX":
                targets = instruction.targets_copy()
                # Process pairs of targets
                for i in range(0, len(targets), 2):
                    if i + 1 < len(targets):
                        control = targets[i]
                        target = targets[i + 1]
                        # Skip if either target is a measurement record (negative value)
                        # or a sweep bit or other non-qubit target
                        if control.is_qubit_target and target.is_qubit_target:
                            new_circuit.append("DEPOLARIZE2", [control.value, target.value], p_cnot)
    
    return new_circuit


# Measurement opcodes that tqec uniform_depolarizing expects with no gate args (assert len(args)==0).
_TQEC_MEASURE_OPS = frozenset({"M", "MX", "MY", "MZ", "MPP"})


def _rewrite_mr_mrx_for_tqec(circuit: stim.Circuit) -> stim.Circuit:
    """Expand MR -> M, TICK, R and MRX -> MX, TICK, RX so tqec uniform_depolarizing has rules for them.
    Also strips gate args from measurement instructions so tqec's NoiseRule (assert len(args)==0) passes."""
    out = stim.Circuit()
    for instruction in circuit:
        if instruction.name == "REPEAT":
            inner = _rewrite_mr_mrx_for_tqec(instruction.body_copy())
            out += inner * instruction.repeat_count
        elif instruction.name == "MR":
            targets = instruction.targets_copy()
            qubits = [t.value for t in targets if t.is_qubit_target]
            out.append("M", qubits)
            out.append("TICK")
            out.append("R", qubits)
        elif instruction.name == "MRX":
            targets = instruction.targets_copy()
            qubits = [t.value for t in targets if t.is_qubit_target]
            out.append("MX", qubits)
            out.append("TICK")
            out.append("RX", qubits)
        elif instruction.name in ("MRY", "MRZ"):
            targets = instruction.targets_copy()
            qubits = [t.value for t in targets if t.is_qubit_target]
            out.append(instruction.name[0] + instruction.name[2], qubits)  # MY, MZ
            out.append("TICK")
            out.append("R" + instruction.name[1], qubits)  # RY, RZ
        else:
            # tqec uniform_depolarizing asserts len(args)==0 for measure ops; strip args if present
            if instruction.name in _TQEC_MEASURE_OPS and instruction.gate_args_copy():
                out.append(instruction.name, instruction.targets_copy())
            else:
                out.append(instruction)
    return out


def apply_noise(circuit: stim.Circuit, p_cnot: float, noise_model: str) -> stim.Circuit:
    """
    Apply a noise model to a noiseless circuit.

    Args:
        circuit: Noiseless stim circuit
        p_cnot: Noise strength (probability parameter)
        noise_model: One of:
            - 'depolarize2_after_cnot': DEPOLARIZE2(p) after each CNOT only
            - 'tqec_uniform_depolarizing': tqec NoiseModel.uniform_depolarizing(p) — idle,
              1q/2q Clifford, measure flips, reset errors

    Returns:
        Noisy circuit
    """
    if noise_model == 'depolarize2_after_cnot':
        return add_depolarize2_after_cnot(circuit, p_cnot)
    if noise_model == 'tqec_uniform_depolarizing':
        from tqec import NoiseModel
        circuit = _rewrite_mr_mrx_for_tqec(circuit)
        return NoiseModel.uniform_depolarizing(p_cnot).noisy_circuit(circuit)
    raise ValueError(f"Unknown noise_model: {noise_model!r}. Use 'depolarize2_after_cnot' or 'tqec_uniform_depolarizing'.")


def generate_gidney_circuit(circuit_type: str, d: int, rounds: int, p_cnot: float) -> stim.Circuit:
    """
    Generate Gidney's circuit using code from Zenodo with custom parameters.
    
    Args:
        circuit_type: 'midout' or 'superdense'
        d: Code distance (base_width for midout, base_data_width for superdense)
        rounds: Number of syndrome extraction rounds
        p_cnot: Unused (kept for API); apply noise via apply_noise(..., noise_model).
        
    Returns:
        Noiseless circuit; caller applies noise via apply_noise(..., noise_model).
    """
    import sys
    if 'gidney_circuits/src' not in sys.path:
        sys.path.insert(0, 'gidney_circuits/src')
    
    import gen
    from clorco.color_code._midout_planar_color_code_circuits import make_midout_color_code_circuit_chunks
    from clorco.color_code._superdense_planar_color_code_circuits import make_superdense_color_code_circuit
    
    if circuit_type == 'midout':
        chunks = make_midout_color_code_circuit_chunks(
            base_width=d,
            basis='X',
            rounds=rounds,
            use_488=False,
        )
        circuit = gen.compile_chunks_into_circuit(chunks)
    elif circuit_type == 'superdense':
        circuit = make_superdense_color_code_circuit(
            base_data_width=d,
            basis='X',
            rounds=rounds,
        )
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    # Noiseless; caller applies noise via apply_noise(..., noise_model)
    return circuit


def build_optimized_circuit(d: int, rounds: int, p_cnot: float, 
                           schedule_config: Dict) -> stim.Circuit:
    """
    Build circuit with optimized schedule from exhaustive search.
    
    Args:
        d: Code distance
        rounds: Number of syndrome extraction rounds
        p_cnot: CNOT error probability
        schedule_config: Dict with 'r_schedule', 'g_schedule', 'b_schedule' keys
                        (6-element Z schedules; X schedules are derived by adding 6)
        
    Returns:
        Noiseless stim.Circuit; caller applies noise via apply_noise(..., noise_model).
    """
    # Edge schedules (fixed, Z part only - same as in exhaustive_schedule_search.py)
    edge_schedules_z = {
        'r': [1, 4, 2, 6, 3, 5],
        'b': [1, 5, 3, 6, 2, 4],
        'g': [4, 2, 6, 3, 5, 1],
    }
    
    # Build full 12-element schedules (Z + X) for both bulk and edge
    # X schedule is derived from Z schedule by adding 6 to each element
    schedules_by_color_and_type = {}
    for color in ['r', 'g', 'b']:
        # Bulk: from config (6 elements) -> full 12 elements
        bulk_z = schedule_config[f'{color}_schedule']
        bulk_full = bulk_z + [x + 6 for x in bulk_z]
        
        # Edge: fixed (6 elements) -> full 12 elements
        edge_z = edge_schedules_z[color]
        edge_full = edge_z + [x + 6 for x in edge_z]
        
        schedules_by_color_and_type[color] = {
            'bulk': bulk_full,
            'edge': edge_full
        }
    
    # Build initial ColorCode to get structure (no noise)
    colorcode_init = ColorCode(d=d, rounds=rounds, cnot_schedule="tri_optimal", 
                               p_cnot=0, exclude_non_essential_pauli_detectors=False)
    
    # Build schedule dict
    cnot_dict = schedule_to_cnot_dict_by_color_and_type(colorcode_init, schedules_by_color_and_type)
    
    # Build ColorCode with this schedule (no noise initially)
    # Use exclude_non_essential_pauli_detectors=False to include X detectors
    colorcode = ColorCode(d=d, rounds=rounds, cnot_schedule=cnot_dict, 
                         p_cnot=0, exclude_non_essential_pauli_detectors=False)
    
    # Noiseless; caller applies noise via apply_noise(..., noise_model)
    return colorcode.circuit


# Schedule config for tri_optimal: same 6-step Z order for all colors (from ColorCode built-in).
# X uses the same pattern at steps 7-12. Used with build_parallel_circuit for tri_optimal circuits.
TRI_OPTIMAL_SCHEDULE_CONFIG: Dict = {
    'r_schedule': [2, 3, 6, 5, 4, 1],
    'g_schedule': [2, 3, 6, 5, 4, 1],
    'b_schedule': [2, 3, 6, 5, 4, 1],
}


def build_tri_optimal_circuit(d: int, rounds: int, p_cnot: float) -> stim.Circuit:
    """
    Build circuit with tri_optimal schedule using the same parallel construction
    as build_parallel_circuit (12 parallel CX steps per round).
    """
    return build_parallel_circuit(d, rounds, p_cnot, TRI_OPTIMAL_SCHEDULE_CONFIG)


def build_parallel_circuit(d: int, rounds: int, p_cnot: float,
                           schedule_config: Dict) -> stim.Circuit:
    """
    Build a parallelized color-code circuit from a schedule config.
    
    Uses manual construction with 12 parallel CX steps per round (6 for Z, 6 for X).
    Suitable for both tri_optimal (pass TRI_OPTIMAL_SCHEDULE_CONFIG) and
    optimized zero-collision schedules (pass config from load_zero_collision_schedule).
    
    Args:
        d: Code distance
        rounds: Number of syndrome extraction rounds
        p_cnot: CNOT error probability
        schedule_config: Dict with 'r_schedule', 'g_schedule', 'b_schedule' keys
                        (each a 6-element list: Z step 1-6 per data position; X uses same at 7-12)
        
    Returns:
        Noiseless stim.Circuit; caller applies noise via apply_noise(..., noise_model).
    """
    # Get ColorCode structure for qubit layout
    colorcode = ColorCode(d=d, rounds=1, cnot_schedule="tri_optimal", p_cnot=0,
                         exclude_non_essential_pauli_detectors=True)
    
    tanner_graph = colorcode.tanner_graph
    anc_Z_qubits = colorcode.qubit_groups['anc_Z']  # reused for X in steps 7-12 after M+RX
    data_qubits = colorcode.qubit_groups['data']
    
    # Canonical offsets for hexagonal plaquettes
    OFFSETS = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
    
    def get_data_qid(anc_qubit, offset_idx):
        """Get data qubit id for an ancilla at given offset."""
        offset = OFFSETS[offset_idx % 6]
        data_x = anc_qubit['x'] + offset[0]
        data_y = anc_qubit['y'] + offset[1]
        data_name = f"{data_x}-{data_y}"
        try:
            return tanner_graph.vs.find(name=data_name).index
        except ValueError:
            return None
    
    # Build full schedules by color (same for bulk and edge in parallel version)
    schedules_by_color = {
        'r': schedule_config['r_schedule'],
        'g': schedule_config['g_schedule'],
        'b': schedule_config['b_schedule'],
    }
    
    # Collect CNOTs grouped by time step (1-12)
    # Z stabilizers: 1-6, X stabilizers: 7-12
    cnots_by_timestep = {t: [] for t in range(1, 13)}
    
    # Z stabilizers
    for anc_qubit in anc_Z_qubits:
        color = anc_qubit['color']
        schedule = schedules_by_color[color]
        anc_qid = anc_qubit.index
        for pos_idx, timestep in enumerate(schedule):
            data_qid = get_data_qid(anc_qubit, pos_idx)
            if data_qid is not None:
                # Z stabilizer: CX data -> ancilla
                cnots_by_timestep[timestep].append((data_qid, anc_qid))
    
    # X stabilizers (time steps 7-12): reuse same ancillas as Z (same plaquette = same support)
    for anc_qubit in anc_Z_qubits:
        color = anc_qubit['color']
        schedule = schedules_by_color[color]
        anc_qid = anc_qubit.index
        for pos_idx, timestep in enumerate(schedule):
            data_qid = get_data_qid(anc_qubit, pos_idx)
            if data_qid is not None:
                # X stabilizer: CX ancilla -> data (at timestep + 6); same qubit as Z ancilla
                cnots_by_timestep[timestep + 6].append((anc_qid, data_qid))
    
    # Qubit IDs: only data + Z ancillas (X ancillas are reused from Z ancillas)
    data_qids = [dq['qid'] for dq in data_qubits]
    anc_qids = [anc['qid'] for anc in anc_Z_qubits]
    num_anc = len(anc_qids)
    used_qids = set(data_qids) | set(anc_qids)
    
    # Build the stim circuit
    circuit = stim.Circuit()
    
    # Add QUBIT_COORDS only for qubits used in the circuit
    for v in tanner_graph.vs:
        if v.index in used_qids:
            circuit.append("QUBIT_COORDS", [v.index], [v['x'], v['y']])
    
    # Initialize data qubits and ancillas (|0⟩ for data and Z ancillas)
    circuit.append("R", data_qids + anc_qids)
    circuit.append("TICK")
    
    def add_syndrome_round(circuit):
        """One round: steps 1-6 (Z), M + RX on ancillas, steps 7-12 (X), MX + R on ancillas. No noise (added later)."""
        # Steps 1-6: Z stabilizer CNOTs (data -> ancilla)
        for t in range(1, 7):
            cx_targets = []
            for ctrl, targ in cnots_by_timestep[t]:
                cx_targets.extend([ctrl, targ])
            if cx_targets:
                circuit.append("CX", cx_targets)
            circuit.append("TICK")
        # Measure Z syndrome, reset ancillas to |+⟩ for X measurement (TICKs so tqec sees separate moments)
        circuit.append("M", anc_qids)
        circuit.append("TICK")
        circuit.append("RX", anc_qids)
        circuit.append("TICK")
        # Steps 7-12: X stabilizer CNOTs (ancilla -> data), same ancilla qubits
        for t in range(7, 13):
            cx_targets = []
            for ctrl, targ in cnots_by_timestep[t]:
                cx_targets.extend([ctrl, targ])
            if cx_targets:
                circuit.append("CX", cx_targets)
            circuit.append("TICK")
        # Measure X syndrome, reset ancillas to |0⟩ for next round (TICKs for tqec)
        circuit.append("MX", anc_qids)
        circuit.append("TICK")
        circuit.append("R", anc_qids)
        circuit.append("TICK")
    
    # First round
    add_syndrome_round(circuit)
    
    # Add detectors for first round (per round: M then MX → 2*num_anc records; Z at -2*num_anc+j, X at -num_anc+j)
    # Z-type: deterministic (data starts in |0⟩)
    for j, anc in enumerate(anc_Z_qubits):
        color_val = 0 if anc['color'] == 'r' else (1 if anc['color'] == 'g' else 2)
        coords = (anc['x'], anc['y'], 0, 2, color_val)
        circuit.append("DETECTOR", [stim.target_rec(-2*num_anc + j)], coords)
    # X-type: NOT deterministic in first round (random outcomes), no detector
    
    circuit.append("SHIFT_COORDS", [], [0, 0, 1])
    
    # Additional rounds
    for round_idx in range(1, rounds):
        add_syndrome_round(circuit)
        
        # Z-type detectors comparing with previous round (current Z at -2*num_anc+j, previous at -4*num_anc+j)
        for j, anc in enumerate(anc_Z_qubits):
            color_val = 0 if anc['color'] == 'r' else (1 if anc['color'] == 'g' else 2)
            coords = (anc['x'], anc['y'], 0, 2, color_val)
            circuit.append("DETECTOR", [
                stim.target_rec(-2*num_anc + j),
                stim.target_rec(-4*num_anc + j)
            ], coords)
        
        # X-type detectors (same ancilla qubits; current X at -num_anc+j, previous at -3*num_anc+j)
        for j, anc in enumerate(anc_Z_qubits):
            color_val = 0 if anc['color'] == 'r' else (1 if anc['color'] == 'g' else 2)
            coords = (anc['x'], anc['y'], 0, 4, color_val)  # 4 = X-type basis
            circuit.append("DETECTOR", [
                stim.target_rec(-num_anc + j),
                stim.target_rec(-3*num_anc + j)
            ], coords)
        
        circuit.append("SHIFT_COORDS", [], [0, 0, 1])
    
    # Final data measurement
    circuit.append("M", data_qids)
    
    # Final detectors comparing data measurements with last round's Z ancilla measurements
    for j, anc in enumerate(anc_Z_qubits):
        color_val = 0 if anc['color'] == 'r' else (1 if anc['color'] == 'g' else 2)
        coords = (anc['x'], anc['y'], 0, 2, color_val)
        
        # Get data qubits connected to this ancilla (same for Z and X)
        connected_data = []
        for pos_idx in range(6):
            data_qid = get_data_qid(anc, pos_idx)
            if data_qid is not None:
                connected_data.append(data_qid)
        
        # Build detector targets: data measurements + last round's Z measurement (at -2*num_anc+j in last round)
        targets = []
        for dq in connected_data:
            targets.append(stim.target_rec(-len(data_qids) + data_qids.index(dq)))
        targets.append(stim.target_rec(-len(data_qids) - 2*num_anc + j))
        
        circuit.append("DETECTOR", targets, coords)
    
    # Observable (top row of data qubits for Z memory)
    top_row_qids = [dq['qid'] for dq in data_qubits if dq['y'] == 0]
    obs_targets = [stim.target_rec(-len(data_qids) + data_qids.index(qid)) for qid in top_row_qids]
    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    
    # Noiseless; caller applies noise via apply_noise(..., noise_model)
    return circuit


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    distances: List[int]
    error_rates: List[float]
    n_shots: int
    decoders: List[str]  # 'tesseract' or 'hyperion'
    circuit_types: List[str]  # 'optimized', 'optimized_parallel', 'tri_optimal', 'midout', 'superdense'
    optimized_schedule: Optional[Dict] = None
    noise_model: str = 'depolarize2_after_cnot'  # or 'tqec_uniform_depolarizing'
    max_errors: Optional[int] = None  # Stop early after this many errors
    max_time_per_config: Optional[float] = None  # Stop early after this many seconds
    rounds: Optional[int] = None  # Number of syndrome extraction rounds. None = use d (distance-dependent)


def compute_circuit_distance_stim(circuit: stim.Circuit, max_exploration_size: int = 4) -> int:
    """
    Compute circuit-level distance using stim's heuristic search.
    
    Args:
        circuit: stim.Circuit to analyze
        max_exploration_size: Maximum detection event set size to explore
        
    Returns:
        Circuit-level distance (weight of minimum undetectable logical error)
    """
    errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=max_exploration_size,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False
    )
    return len(errors)


def run_sinter_single_task(
    circuit: stim.Circuit,
    decoder: str,
    metadata: dict,
    max_shots: int,
    max_errors: int,
    num_workers: int,
) -> dict:
    """
    Run sinter for a single circuit + decoder combination.
    
    Args:
        circuit: The stim circuit to benchmark
        decoder: Decoder name ('tesseract' or 'hyperion')
        metadata: Dictionary with configuration metadata
        max_shots: Maximum shots per task
        max_errors: Maximum errors for early stopping
        num_workers: Number of parallel workers
        
    Returns:
        Dict with results
    """
    from scipy import stats
    
    task = sinter.Task(
        circuit=circuit,
        decoder=decoder,
        json_metadata=metadata,
    )
    
    start_time = time.time()
    results = sinter.collect(
        tasks=[task],
        max_shots=max_shots,
        max_errors=max_errors,
        num_workers=num_workers,
        custom_decoders=CUSTOM_SINTER_DECODERS,
    )
    decode_time = time.time() - start_time
    
    # Process result
    if results:
        sample = results[0]
        shots = sample.shots
        errors = sample.errors
        
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
        
        return {
            'shots_used': shots,
            'errors': errors,
            'error_rate': error_rate,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'decode_time': decode_time,
        }
    
    return None


def run_benchmark_sinter(config: BenchmarkConfig, save_path: str = None,
                         num_workers: int = 8) -> pd.DataFrame:
    """
    Run benchmark using Sinter for efficient parallel decoding.
    
    Runs sinter separately for each configuration (circuit_type, distance, 
    error_rate, decoder) and saves results incrementally.
    
    Args:
        config: BenchmarkConfig with parameters
        save_path: Optional path to save results incrementally
        num_workers: Number of parallel workers for Sinter
    
    Returns DataFrame with results.
    """
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # Force unbuffered output
    
    results = []
    circuit_distances = {}  # Track circuit-level distances
    
    # Count total configurations
    total_configs = (len(config.circuit_types) * len(config.distances) * 
                    len(config.error_rates) * len(config.decoders))
    current_config = 0
    
    print("=" * 70)
    print("Running Sinter Benchmark")
    print("=" * 70)
    print(f"Total configurations: {total_configs}")
    print(f"Max shots: {config.n_shots:,}")
    print(f"Max errors: {config.max_errors}")
    print(f"Num workers: {num_workers}")
    print("=" * 70)
    
    # Initialize CSV with header if save_path provided
    if save_path:
        csv_path = save_path.replace('.pkl', '.csv')
        fieldnames = ['distance', 'rounds', 'p_cnot', 'circuit_type', 'decoder',
                     'circuit_distance', 'n_shots', 'shots_used', 'errors',
                     'error_rate', 'ci_low', 'ci_high', 'decode_time']
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Loop: distance -> circuit_type -> decoder -> error_rate
    for d in config.distances:
        rounds = config.rounds if config.rounds is not None else d
        print(f"\n{'='*70}")
        print(f"DISTANCE: d={d}, rounds={rounds}")
        print(f"{'='*70}")
        
        for circuit_type in config.circuit_types:
            print(f"\n  Circuit type: {circuit_type}")
            
            # Compute circuit distance once per (circuit_type, d)
            distance_key = (circuit_type, d)
            if distance_key not in circuit_distances:
                # Build circuit with noise so Stim's search has error channels to explore
                try:
                    if circuit_type == 'optimized':
                        if config.optimized_schedule is None:
                            continue
                        test_circuit = build_optimized_circuit(d, rounds, 0.001, config.optimized_schedule)
                    elif circuit_type == 'optimized_parallel':
                        if config.optimized_schedule is None:
                            continue
                        test_circuit = build_parallel_circuit(d, rounds, 0.001, config.optimized_schedule)
                    elif circuit_type == 'tri_optimal':
                        test_circuit = build_tri_optimal_circuit(d, rounds, 0.001)
                    elif circuit_type in ['midout', 'superdense']:
                        test_circuit = generate_gidney_circuit(circuit_type, d, rounds, 0.001)
                    else:
                        continue
                    # Apply noise so detector error model has errors; search needs them to find undetectable logical errors
                    test_circuit = apply_noise(test_circuit, 0.001, config.noise_model)
                    circuit_distances[distance_key] = compute_circuit_distance_stim(test_circuit)
                    print(f"    Circuit-level distance: {circuit_distances[distance_key]}")
                except Exception as e:
                    print(f"    Error computing distance: {e}")
                    circuit_distances[distance_key] = None
            
            for decoder in config.decoders:
                for p_cnot in config.error_rates:
                    current_config += 1
                    
                    # Build circuit
                    try:
                        if circuit_type == 'optimized':
                            if config.optimized_schedule is None:
                                print(f"      Skipping (no schedule)")
                                continue
                            circuit = build_optimized_circuit(d, rounds, p_cnot, config.optimized_schedule)
                        elif circuit_type == 'optimized_parallel':
                            if config.optimized_schedule is None:
                                print(f"      Skipping (no schedule)")
                                continue
                            circuit = build_parallel_circuit(d, rounds, p_cnot, config.optimized_schedule)
                        elif circuit_type == 'tri_optimal':
                            circuit = build_tri_optimal_circuit(d, rounds, p_cnot)
                        elif circuit_type in ['midout', 'superdense']:
                            circuit = generate_gidney_circuit(circuit_type, d, rounds, p_cnot)
                        else:
                            continue
                        circuit = apply_noise(circuit, p_cnot, config.noise_model)
                    except Exception as e:
                        print(f"      [{current_config}/{total_configs}] {decoder}, p={p_cnot:.0e}: ERROR building circuit: {e}")
                        continue
                    
                    print(f"      [{current_config}/{total_configs}] {decoder}, p={p_cnot:.0e}: ", end='', flush=True)
                    
                    metadata = {
                        'distance': d,
                        'rounds': rounds,
                        'p_cnot': p_cnot,
                        'circuit_type': circuit_type,
                        'decoder': decoder,
                    }
                    
                    try:
                        result = run_sinter_single_task(
                            circuit=circuit,
                            decoder=decoder,
                            metadata=metadata,
                            max_shots=config.n_shots,
                            max_errors=config.max_errors,
                            num_workers=num_workers,
                        )
                        
                        if result:
                            row = {
                                'distance': d,
                                'rounds': rounds,
                                'p_cnot': p_cnot,
                                'circuit_type': circuit_type,
                                'decoder': decoder,
                                'circuit_distance': circuit_distances.get(distance_key),
                                'n_shots': config.n_shots,
                                **result,
                            }
                            results.append(row)
                            
                            print(f"{result['errors']}/{result['shots_used']} errors, "
                                  f"rate={result['error_rate']:.6f}, time={result['decode_time']:.1f}s")
                            
                            # Save incrementally
                            if save_path:
                                import csv
                                with open(csv_path, 'a', newline='') as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    writer.writerow(row)
                        else:
                            print("No result")
                            
                    except Exception as e:
                        print(f"ERROR: {e}")
    
    df = pd.DataFrame(results)
    
    # Final save
    if save_path and not df.empty:
        df.to_pickle(save_path)
        print(f"\nResults saved to {save_path}")
    
    return df


def load_zero_collision_schedule(csv_file: str = 'results/zero_collision_schedules.csv', 
                                  index: int = 0) -> Dict:
    """
    Load a zero-collision schedule from the CSV file.
    
    These schedules have been verified to have:
    - Zero CNOT collisions between plaquettes (parallelizable)
    - Circuit-level distance 6 (preserved from original)
    
    Args:
        csv_file: Path to zero_collision_schedules.csv
        index: Which schedule to load (0 = first, default)
        
    Returns:
        dict with 'r_schedule', 'g_schedule', 'b_schedule' keys
    """
    import csv
    import ast
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == index:
                return {
                    'r_schedule': ast.literal_eval(row['r_schedule']),
                    'g_schedule': ast.literal_eval(row['g_schedule']),
                    'b_schedule': ast.literal_eval(row['b_schedule']),
                }
    
    raise ValueError(f"Schedule index {index} not found in {csv_file}")


def generate_circuit_links_html(d: int, rounds: int, p_cnot: float,
                                optimized_schedule: Optional[Dict] = None,
                                output_path: str = 'results/circuit_links.html',
                                noise_model: str = 'depolarize2_after_cnot'):
    """
    Generate HTML file with Crumble links for all circuit types.
    
    Args:
        d: Code distance
        rounds: Number of syndrome extraction rounds
        p_cnot: CNOT error probability
        optimized_schedule: Dict with schedule for optimized circuits
        output_path: Path to save HTML file
        noise_model: 'depolarize2_after_cnot' or 'tqec_uniform_depolarizing'
    """
    import urllib.parse
    
    circuits = {}
    
    def _build_and_noise(build_fn, *args, **kwargs):
        circ = build_fn(*args, **kwargs)
        return apply_noise(circ, p_cnot, noise_model)

    # Build all circuit types (noiseless then apply_noise)
    if optimized_schedule:
        try:
            circuits['Optimized (serialized)'] = _build_and_noise(build_optimized_circuit, d, rounds, p_cnot, optimized_schedule)
        except Exception as e:
            import traceback
            print(f"Error building optimized circuit: {e}")
            traceback.print_exc()
        
        try:
            circuits['Optimized (parallel)'] = _build_and_noise(build_parallel_circuit, d, rounds, p_cnot, optimized_schedule)
        except Exception as e:
            print(f"Error building optimized_parallel circuit: {e}")
    
    try:
        circuits['Tri-Optimal'] = _build_and_noise(build_tri_optimal_circuit, d, rounds, p_cnot)
    except Exception as e:
        print(f"Error building tri_optimal circuit: {e}")
    
    try:
        circuits['Midout (Gidney)'] = _build_and_noise(generate_gidney_circuit, 'midout', d, rounds, p_cnot)
    except Exception as e:
        print(f"Error building midout circuit: {e}")
    
    try:
        circuits['Superdense (Gidney)'] = _build_and_noise(generate_gidney_circuit, 'superdense', d, rounds, p_cnot)
    except Exception as e:
        print(f"Error building superdense circuit: {e}")
    
    # Generate HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Circuit Validation - Crumble Links</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .circuit {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .circuit h2 {{ margin-top: 0; color: #2c3e50; }}
        .stats {{ color: #666; margin-bottom: 10px; }}
        .schedule {{ color: #888; font-family: monospace; font-size: 12px; margin-bottom: 10px; }}
        a {{ color: #3498db; text-decoration: none; font-weight: bold; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Circuit Validation Links (d={d}, rounds={rounds}, p={p_cnot})</h1>
'''
    
    for name, circuit in circuits.items():
        # Get circuit stats
        num_qubits = circuit.num_qubits
        num_detectors = circuit.num_detectors
        
        # Count CX gates per tick to show parallelization
        cx_per_tick = {}
        tick = 0
        for inst in circuit.flattened():
            if inst.name == "TICK":
                tick += 1
            elif inst.name == "CX":
                n = len(inst.targets_copy()) // 2
                cx_per_tick[tick] = cx_per_tick.get(tick, 0) + n
        
        total_cx_ticks = len([t for t, c in cx_per_tick.items() if c > 0])
        max_parallel_cx = max(cx_per_tick.values()) if cx_per_tick else 0
        
        # Generate Crumble URL
        crumble_url = "https://algassert.com/crumble#circuit=" + urllib.parse.quote(str(circuit), safe='')
        
        # Add schedule info for optimized circuits
        schedule_info = ""
        if optimized_schedule and "Optimized" in name:
            schedule_info = f'''
        <p class="schedule">Schedule: R={optimized_schedule['r_schedule']}, G={optimized_schedule['g_schedule']}, B={optimized_schedule['b_schedule']}</p>'''
        
        html += f'''
    <div class="circuit">
        <h2>{name}</h2>
        <p class="stats">Qubits: {num_qubits} | Detectors: {num_detectors} | CX ticks: {total_cx_ticks} | Max parallel CX: {max_parallel_cx}</p>{schedule_info}
        <a href="{crumble_url}" target="_blank">Open in Crumble →</a>
    </div>
'''
    
    html += '''
</body>
</html>'''
    
    # Save HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Saved circuit links to {output_path}")
    return circuits


# =============================================================================
# Plotting
# =============================================================================

def plot_threshold(df: pd.DataFrame, decoder: str, save_path: str = None):
    """
    Plot threshold curves: logical error rate vs physical error rate.
    
    Args:
        df: DataFrame with benchmark results
        decoder: Decoder to plot ('tesseract' or 'hyperion')
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    # Filter by decoder
    df_decoder = df[df['decoder'] == decoder].copy()
    
    if df_decoder.empty:
        print(f"No data for decoder: {decoder}")
        return
    
    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    circuit_types = df_decoder['circuit_type'].unique()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df_decoder['distance'].unique())))
    
    for ax_idx, circuit_type in enumerate(circuit_types):
        if ax_idx >= 4:
            break
        ax = axes[ax_idx]
        df_circuit = df_decoder[df_decoder['circuit_type'] == circuit_type]
        
        for color_idx, d in enumerate(sorted(df_circuit['distance'].unique())):
            df_d = df_circuit[df_circuit['distance'] == d].sort_values('p_cnot')
            
            ax.errorbar(
                df_d['p_cnot'], 
                df_d['error_rate'],
                yerr=[df_d['error_rate'] - df_d['ci_low'], 
                      df_d['ci_high'] - df_d['error_rate']],
                marker='o', 
                label=f'd={d}',
                color=colors[color_idx],
                capsize=3,
            )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical error rate (p)')
        ax.set_ylabel('Logical error rate')
        ax.set_title(f'{circuit_type} ({decoder})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved threshold plot to {save_path}")
    
    plt.show()


def plot_comparison(df: pd.DataFrame, decoder: str, p_target: float = 0.001, 
                   save_path: str = None):
    """
    Plot comparison of all circuit types at a fixed physical error rate.
    
    Args:
        df: DataFrame with benchmark results
        decoder: Decoder to plot
        p_target: Target physical error rate
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    # Filter by decoder and find closest p to target
    df_decoder = df[df['decoder'] == decoder].copy()
    
    if df_decoder.empty:
        print(f"No data for decoder: {decoder}")
        return
    
    # Find closest p to target
    available_p = df_decoder['p_cnot'].unique()
    closest_p = available_p[np.argmin(np.abs(available_p - p_target))]
    
    df_p = df_decoder[df_decoder['p_cnot'] == closest_p]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    circuit_types = sorted(df_p['circuit_type'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for idx, circuit_type in enumerate(circuit_types):
        df_circuit = df_p[df_p['circuit_type'] == circuit_type].sort_values('distance')
        
        ax.errorbar(
            df_circuit['distance'], 
            df_circuit['error_rate'],
            yerr=[df_circuit['error_rate'] - df_circuit['ci_low'], 
                  df_circuit['ci_high'] - df_circuit['error_rate']],
            marker=markers[idx % len(markers)], 
            label=circuit_type,
            color=colors[idx % len(colors)],
            capsize=3,
            markersize=8,
            linewidth=2,
        )
    
    ax.set_yscale('log')
    ax.set_xlabel('Code distance (d)', fontsize=12)
    ax.set_ylabel('Logical error rate', fontsize=12)
    ax.set_title(f'Circuit Comparison ({decoder}, p={closest_p:.0e})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df_p['distance'].unique()))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def compute_qubit_count(d: int, circuit_type: str) -> int:
    """
    Compute total qubit count for a circuit type at given distance.
    
    Qubit count = data_qubits + auxiliaries_per_plaquette * num_plaquettes
    
    For triangular color code of distance d:
    - data_qubits = (3*d^2 + 1) / 4
    - num_plaquettes = 3*(d^2 - 1) / 8
    
    Auxiliaries per plaquette:
    - superdense: 2 (one Z ancilla, one X ancilla per plaquette)
    - tri_optimal, optimized_parallel: 1 (shared ancilla for Z and X)
    - midout: 0 (no dedicated ancillas, uses data qubits)
    
    Args:
        d: Code distance
        circuit_type: Type of circuit
        
    Returns:
        Total qubit count
    """
    # Data qubits for triangular color code
    data_qubits = (3 * d * d + 1) // 4
    
    # Number of plaquettes (stabilizers of each type)
    num_plaquettes = 3 * (d * d - 1) // 8
    
    # Auxiliaries per plaquette based on circuit type
    if circuit_type == 'superdense':
        aux_per_plaquette = 2
    elif circuit_type in ['tri_optimal', 'optimized_parallel', 'optimized']:
        aux_per_plaquette = 1
    elif circuit_type == 'midout':
        aux_per_plaquette = 0
    else:
        aux_per_plaquette = 1  # default
    
    return data_qubits + aux_per_plaquette * num_plaquettes


def plot_error_vs_qubits(df: pd.DataFrame, decoder: str = None, 
                         save_path: str = None):
    """
    Plot logical error rate per round vs number of qubits (sqrt scale on x-axis).
    
    Creates one subplot for each error rate in the data.
    X-axis shows actual qubit counts but with sqrt spacing.
    Y-axis shows logical error rate divided by number of rounds.
    
    Args:
        df: DataFrame with benchmark results (must have 'rounds' column)
        decoder: Decoder to plot (if None, uses first available)
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    # Filter by decoder if specified
    if decoder is not None:
        df_plot = df[df['decoder'] == decoder].copy()
    else:
        df_plot = df.copy()
        decoder = df_plot['decoder'].iloc[0] if not df_plot.empty else 'unknown'
    
    if df_plot.empty:
        print(f"No data for decoder: {decoder}")
        return
    
    # Add qubit count column
    df_plot['num_qubits'] = df_plot.apply(
        lambda row: compute_qubit_count(row['distance'], row['circuit_type']), 
        axis=1
    )
    
    # Add per-round error rate columns
    df_plot['error_rate_per_round'] = df_plot['error_rate'] / df_plot['rounds']
    df_plot['ci_low_per_round'] = df_plot['ci_low'] / df_plot['rounds']
    df_plot['ci_high_per_round'] = df_plot['ci_high'] / df_plot['rounds']
    
    # Get unique error rates
    error_rates = sorted(df_plot['p_cnot'].unique())
    n_plots = len(error_rates)
    
    if n_plots == 0:
        print("No error rates found in data")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes.flatten()
    
    # Color and marker maps for circuit types
    circuit_colors = {
        'optimized_parallel': '#2ecc71',  # green
        'tri_optimal': '#3498db',         # blue
        'midout': '#e74c3c',              # red
        'superdense': '#9b59b6',          # purple
        'optimized': '#f39c12',           # orange
    }
    circuit_markers = {
        'optimized_parallel': 'o',
        'tri_optimal': 's',
        'midout': '^',
        'superdense': 'D',
        'optimized': 'p',
    }
    
    for ax_idx, p_cnot in enumerate(error_rates):
        ax = axes[ax_idx]
        df_p = df_plot[df_plot['p_cnot'] == p_cnot]
        
        # Get rounds value for title
        rounds_val = df_p['rounds'].iloc[0] if 'rounds' in df_p.columns else 'N/A'
        
        for circuit_type in sorted(df_p['circuit_type'].unique()):
            df_ct = df_p[df_p['circuit_type'] == circuit_type].sort_values('num_qubits')
            
            color = circuit_colors.get(circuit_type, '#333333')
            marker = circuit_markers.get(circuit_type, 'o')
            
            # Handle error bars (some might be None) - use per-round values
            yerr_low = df_ct['error_rate_per_round'] - df_ct['ci_low_per_round'].fillna(0)
            yerr_high = df_ct['ci_high_per_round'].fillna(df_ct['error_rate_per_round']) - df_ct['error_rate_per_round']
            
            ax.errorbar(
                df_ct['num_qubits'], 
                df_ct['error_rate_per_round'],
                yerr=[yerr_low.clip(lower=0), yerr_high.clip(lower=0)],
                marker=marker, 
                label=circuit_type,
                color=color,
                capsize=3,
                markersize=8,
                linewidth=2,
            )
            
            # Add distance labels next to points
            for _, row in df_ct.iterrows():
                ax.annotate(
                    f'd={int(row["distance"])}',
                    (row['num_qubits'], row['error_rate_per_round']),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.7,
                )
        
        # Set sqrt scale on x-axis (shows actual qubit counts, spaced by sqrt)
        ax.set_xscale('function', functions=(np.sqrt, np.square))
        ax.set_yscale('log')
        
        # Set x-axis ticks to show actual qubit counts
        all_qubits = sorted(df_p['num_qubits'].unique())
        ax.set_xticks(all_qubits)
        ax.set_xticklabels([str(int(q)) for q in all_qubits])
        
        ax.set_xlabel('Number of qubits (sqrt scale)', fontsize=12)
        ax.set_ylabel('Logical error rate (per round)', fontsize=12)
        ax.set_title(f'p = {p_cnot:.4f}, rounds = {rounds_val}', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits with some padding
        x_min = df_p['num_qubits'].min() * 0.8
        x_max = df_p['num_qubits'].max() * 1.1
        ax.set_xlim(x_min, x_max)
    
    plt.suptitle(f'Logical Error Rate per Round vs Qubit Count ({decoder})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved qubit scaling plot to {save_path}")
    
    plt.show()
    
    # Print qubit counts for reference
    print("\nQubit counts by circuit type and distance:")
    for circuit_type in df_plot['circuit_type'].unique():
        print(f"  {circuit_type}:")
        for d in sorted(df_plot['distance'].unique()):
            qubits = compute_qubit_count(d, circuit_type)
            print(f"    d={d}: {qubits} qubits (sqrt={np.sqrt(qubits):.2f})")


def plot_all_results(df: pd.DataFrame, output_dir: str = 'results'):
    """
    Generate all plots from benchmark results.
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for decoder in df['decoder'].unique():
        # Threshold plots
        plot_threshold(df, decoder, 
                      save_path=f'{output_dir}/threshold_{decoder}.png')
        
        # Comparison plots at different error rates
        for p in [0.001, 0.003, 0.005]:
            plot_comparison(df, decoder, p_target=p,
                           save_path=f'{output_dir}/comparison_{decoder}_p{p:.0e}.png')
        
        # Qubit scaling plot
        plot_error_vs_qubits(df, decoder,
                            save_path=f'{output_dir}/qubits_scaling_{decoder}.png')


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the full benchmark comparison."""
    
    # Configuration
    distances = [3,7]#[3, 5, 7, 9, 11]
    error_rates = [0.001]  # 4 key points
    n_shots = 10_000_000  # Max shots
    max_errors = 300  # Stop early after 300 errors
    max_time_per_config = 300
    rounds = None  # Constant number of rounds (set to None for rounds = d)
    decoders = ['tesseract']  # Tesseract + Hyperion (MWPF)
    noise_model = 'tqec_uniform_depolarizing'  # 'depolarize2_after_cnot' or 'tqec_uniform_depolarizing'
    circuit_types = ['optimized_parallel', 'tri_optimal', 'midout', 'superdense']
    save_path = 'results/benchmark_results_0.001_temp_blah.pkl'
    
    # Load zero-collision schedule (for parallelized circuit)
    try:
        optimized_schedule = load_zero_collision_schedule('results/zero_collision_schedules.csv', index=0)
        print(f"Loaded zero-collision schedule (parallelizable, distance=6):")
        print(f"  R: {optimized_schedule['r_schedule']}")
        print(f"  G: {optimized_schedule['g_schedule']}")
        print(f"  B: {optimized_schedule['b_schedule']}")
    except FileNotFoundError:
        print("Warning: zero_collision_schedules.csv not found. Skipping optimized schedule.")
        optimized_schedule = None
        circuit_types = ['tri_optimal', 'midout', 'superdense']
    
    # Generate circuit links HTML for visualization
    print("\n" + "=" * 80)
    print("GENERATING CIRCUIT LINKS")
    print("=" * 80)
    generate_circuit_links_html(
        d=5, rounds=rounds if rounds is not None else 5, p_cnot=0.001,
        optimized_schedule=optimized_schedule,
        output_path='results/circuit_links.html',
        noise_model=noise_model,
    )
    
    # Create benchmark config
    config = BenchmarkConfig(
        distances=distances,
        error_rates=error_rates,
        n_shots=n_shots,
        decoders=decoders,
        circuit_types=circuit_types,
        optimized_schedule=optimized_schedule,
        noise_model=noise_model,
        max_errors=max_errors,
        max_time_per_config=max_time_per_config,
        rounds=rounds,
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Distances: {distances}")
    print(f"Rounds: {rounds if rounds is not None else 'd (distance-dependent)'}")
    print(f"Error rates: {len(error_rates)} points from {error_rates[0]:.0e} to {error_rates[-1]:.0e}")
    print(f"Max shots per config: {n_shots:,}")
    print(f"Max errors per config: {max_errors} (early stopping)")
    print(f"Decoders: {decoders}")
    print(f"Circuit types: {circuit_types}")
    print(f"Noise model: {config.noise_model}")
    print(f"Total configurations: {len(distances) * len(error_rates) * len(circuit_types) * len(decoders)}")
    print("=" * 80 + "\n")
    
    # Run benchmark using Sinter for efficient parallel decoding
    os.makedirs('results', exist_ok=True)
    num_workers = 8
    results_df = run_benchmark_sinter(
        config, 
        save_path=save_path,
        num_workers=num_workers
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nTotal configurations run: {len(results_df)}")
    print(f"Successful: {results_df['error_rate'].notna().sum()}")
    print(f"Failed: {results_df['error_rate'].isna().sum()}")
    
    # Quick summary by circuit type and decoder
    print("\nAverage error rates by circuit type and decoder (p=0.001, d=7):")
    subset = results_df[(results_df['p_cnot'].between(0.0009, 0.0011)) & 
                        (results_df['distance'] == 7)]
    if not subset.empty:
        summary = subset.groupby(['circuit_type', 'decoder'])['error_rate'].mean()
        print(summary.to_string())
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_all_results(results_df, output_dir='results')
    
    return results_df


if __name__ == "__main__":
    main()

