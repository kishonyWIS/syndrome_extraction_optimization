import stim
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any, Union
from collections import defaultdict
import random
import pymatching
from stim_circuit_builder import build_memory_experiment_circuit
from rotated_surface_code import RotatedSurfaceCode
import matplotlib.pyplot as plt

def analyze_circuit_errors(circuit: stim.Circuit, 
                         n_shots: int = 10000) -> Tuple[float, Dict[int, float]]:
    """
    Analyzes error patterns in a memory experiment circuit and identifies problematic qubits.
    Uses PyMatching to decode syndromes before computing logical error rate.
    
    Args:
        circuit: The memory experiment circuit
        n_shots: Number of shots to sample
        
    Returns:
        Tuple of:
        - Logical error rate (after decoding)
        - Dictionary mapping qubit indices to their error involvement frequency
    """
    # Get detector error model and compile sampler
    dem = circuit.detector_error_model()
    sampler = dem.compile_sampler()
    
    try:
        # Create decoder from detector error model
        matching = pymatching.Matching.from_detector_error_model(dem)
        
        # Sample with error tracking
        sample_results: Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]] = sampler.sample(shots=n_shots, return_errors=True)
        dets, obs, errs = sample_results
        
        if errs is None:
            print("No error information returned from sampler")
            return 0.0, {}
            
        # Get circuit error explanations
        circuit_errors = circuit.explain_detector_error_model_errors()
            
        # Decode each shot
        predicted_observables = matching.decode_batch(dets)
        
        # Check if any logical operator was wrongly corrected
        # For multiple observables, check if any observable was wrongly corrected
        logical_errors = np.logical_xor(predicted_observables, obs).any(axis=1)  # Any observable wrong
        logical_error_rate = float(np.mean(logical_errors))
        print(f"Logical error rate (after decoding): {logical_error_rate:.3f}")
        
        # Track qubit involvement in failed shots
        qubit_involvement = defaultdict(int)
        n_failed_shots = 0
        
        # Get indices of shots with logical errors
        error_shot_indices = np.where(logical_errors)[0]
        n_failed_shots = len(error_shot_indices)
        
        # Analyze each shot with a logical error (after decoding)
        for shot_idx in error_shot_indices:
            # Get the error mechanisms for this shot
            shot_errors = errs[shot_idx]
            
            # For each error that occurred
            for error_idx in np.where(shot_errors)[0]:
                explanation = circuit_errors[error_idx]
                # Find the set of qubits that appear as targets in ALL circuit_error_locations
                if explanation.circuit_error_locations:
                    # For each location, get the set of qubit indices in flipped_pauli_product
                    sets_of_qubits = [
                        set(target.gate_target.value for target in loc.flipped_pauli_product)
                        for loc in explanation.circuit_error_locations
                    ]
                    # Find intersection (qubits present in all locations)
                    if sets_of_qubits:
                        common_qubits = set.intersection(*sets_of_qubits)
                        for qubit_idx in common_qubits:
                            qubit_involvement[qubit_idx] += 1
        
        # Convert counts to frequencies
        qubit_frequencies = {
            q: count / max(n_failed_shots, 1)  # Avoid division by zero
            for q, count in qubit_involvement.items()
        }
        
        return logical_error_rate, qubit_frequencies
        
    except Exception as e:
        print(f"Error during sampling or decoding: {e}")
        return 0.0, {}

def optimize_cx_order(css_code, problematic_ancilla: int):
    """
    Creates a new random CX order for the stabilizer containing the problematic ancilla.
    
    Args:
        css_code: The CSS code object
        problematic_ancilla: Index of the problematic ancilla qubit
        stabilizer_type: 'X' or 'Z' indicating stabilizer type
    """
    # Find which stabilizer this ancilla belongs to
    stab_type, stab_idx = css_code.ancilla_index_to_stabilizer(problematic_ancilla)        
    # Get current order and create a new random permutation
    current_order = css_code.get_cx_order(stab_type, stab_idx)
    new_order = list(current_order)
    random.shuffle(new_order)
    css_code.set_cx_order(stab_type, stab_idx, new_order)


def optimize_circuit_cx_orders(css_code, n_shots=10000, noise_prob=0.01, n_steps=100, n_rounds=3) -> Optional[List[Tuple[float, int, List[int]]]]:
    """
    Main optimization loop that:
    1. Builds circuit
    2. Analyzes error patterns (using decoding)
    3. Identifies worst ancilla
    4. Optimizes its CX order
    Repeats for n_steps.
    
    Returns:
        Optional list of (error_rate, worst_ancilla_idx, new_cx_order) for each step
    """
    history = []
    for step in range(n_steps):
        print(f"\n--- Optimization step {step+1}/{n_steps} ---")
        # Build circuit
        circuit = build_memory_experiment_circuit(css_code, n_rounds=n_rounds, noise_prob=noise_prob)
        # Analyze errors (with decoding)
        error_rate, qubit_freqs = analyze_circuit_errors(circuit, n_shots)
        if not qubit_freqs:
            print("No problematic qubits found")
            break
        # Find most problematic ancilla among the qubits
        ancilla_freqs = {q: freq for q, freq in qubit_freqs.items() 
                        if q in css_code.z_ancilla_indices or q in css_code.x_ancilla_indices}
        if not ancilla_freqs:
            print("No ancilla qubits involved in errors. Stopping optimization.")
            break
        worst_ancilla = max(ancilla_freqs.items(), key=lambda x: x[1])
        print(f"Most problematic ancilla: {worst_ancilla[0]} (involved in {worst_ancilla[1]:.1%} of logical errors)")
        # Determine if it's an X or Z stabilizer ancilla and optimize its CX order
        stab_type, stab_idx = css_code.ancilla_index_to_stabilizer(worst_ancilla[0])
        optimize_cx_order(css_code, worst_ancilla[0])
        new_cx_order = css_code.get_cx_order(stab_type, stab_idx)
        history.append((error_rate, worst_ancilla[0], new_cx_order))
    if not history:
        return None
    return history 