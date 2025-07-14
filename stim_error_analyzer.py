import stim
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any, Union, Callable
from collections import defaultdict
import random
import pymatching
from stim_circuit_builder import build_memory_experiment_circuit
from rotated_surface_code import RotatedSurfaceCode
import matplotlib.pyplot as plt
from scipy import stats

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score interval for binomial proportion.
    
    Args:
        successes: Number of successful events (logical errors in our case)
        total: Total number of trials (shots)
        confidence: Confidence level (default 0.95 for 95% confidence)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf((1 + confidence) / 2)
    p_hat = successes / total
    
    # Wilson score interval formula
    denominator = 1 + z**2 / total
    centre_adjusted_probability = float((p_hat + z * z / (2 * total)) / denominator)
    adjusted_standard_error = float(z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denominator)
    
    lower_bound = max(0.0, centre_adjusted_probability - adjusted_standard_error)
    upper_bound = min(1.0, centre_adjusted_probability + adjusted_standard_error)
    
    return (lower_bound, upper_bound)

def analyze_circuit_errors_unified(circuit: stim.Circuit,
                                 decoder: Callable[[np.ndarray], Union[np.ndarray, Tuple[np.ndarray, Any]]],
                                 n_shots: int = 10000) -> Tuple[float, Tuple[float, float], Dict[int, float]]:
    """
    Unified error analysis function that works with any decoder.
    
    Args:
        circuit: The memory experiment circuit
        decoder: Function that takes detector measurements and returns predicted observables
        n_shots: Number of shots to sample
        
    Returns:
        Tuple of:
        - Logical error rate (after decoding)
        - Confidence interval (lower, upper) for the error rate
        - Dictionary mapping qubit indices to their error involvement frequency
    """
    # Get detector error model and compile sampler
    dem = circuit.detector_error_model()
    
    # List out the errors, for later lookup
    error_instructions = []
    for instruction in dem.flattened():
        if instruction.type == 'error':
            error_instructions.append(instruction)
    
    sampler = dem.compile_sampler()
    
    try:
        # Sample with error tracking
        sample_results: Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]] = sampler.sample(shots=n_shots, return_errors=True)
        dets, obs, errs = sample_results
        
        if errs is None:
            print("No error information returned from sampler")
            return 0.0, (0.0, 0.0), {}
            
        # Decode each shot using the provided decoder
        predicted_observables = decoder(dets)
        
        # Handle the return type - decode returns either ndarray or Tuple[ndarray, dict]
        if isinstance(predicted_observables, tuple):
            predicted_observables = predicted_observables[0]  # Extract the array from the tuple
        
        # Ensure predicted_observables has the right shape for comparison
        if predicted_observables.ndim == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        # Check if any logical operator was wrongly corrected
        # For multiple observables, check if any observable was wrongly corrected
        logical_errors = np.logical_xor(predicted_observables, obs).any(axis=1)  # Any observable wrong
        n_errors = int(np.sum(logical_errors))
        logical_error_rate = float(np.mean(logical_errors))
        
        # Calculate confidence interval
        confidence_interval = wilson_score_interval(n_errors, n_shots)
        
        print(f"Logical error rate (after decoding): {logical_error_rate:.6f} [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
        
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
            
            # Create a filter DEM containing only the errors that occurred in this shot
            dem_filter = stim.DetectorErrorModel()
            for error_index in np.flatnonzero(shot_errors):
                dem_filter.append(error_instructions[error_index])
            
            # Get circuit error explanations only for this shot's errors
            circuit_errors = circuit.explain_detector_error_model_errors(
                dem_filter=dem_filter,
            )
            
            # For each error explanation (corresponds to errors in dem_filter)
            for explanation in circuit_errors:
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
        
        return logical_error_rate, confidence_interval, qubit_frequencies
        
    except Exception as e:
        print(f"Error during sampling or decoding: {e}")
        return 0.0, (0.0, 0.0), {}

def analyze_circuit_errors(circuit: stim.Circuit,
                         n_shots: int = 10000) -> Tuple[float, Tuple[float, float], Dict[int, float]]:
    """
    Analyzes error patterns in a memory experiment circuit and identifies problematic qubits.
    Uses PyMatching to decode syndromes before computing logical error rate.
    
    Args:
        circuit: The memory experiment circuit
        n_shots: Number of shots to sample
        
    Returns:
        Tuple of:
        - Logical error rate (after decoding)
        - Confidence interval (lower, upper) for the error rate
        - Dictionary mapping qubit indices to their error involvement frequency
    """
    # Create decoder from detector error model
    dem = circuit.detector_error_model()
    decoder = pymatching.Matching.from_detector_error_model(dem)
    
    # Use the unified function with PyMatching decoder
    return analyze_circuit_errors_unified(circuit, decoder.decode_batch, n_shots)

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


def optimize_circuit_cx_orders(css_code, n_shots=10000, noise_prob=0.01, n_steps=100, n_rounds=3) -> Optional[List[Tuple[float, Tuple[float, float], int, List[int]]]]:
    """
    Main optimization loop that:
    1. Builds circuit
    2. Analyzes error patterns (using decoding)
    3. Identifies worst ancilla
    4. Optimizes its CX order
    Repeats for n_steps.
    
    Returns:
        Optional list of (error_rate, confidence_interval, worst_ancilla_idx, new_cx_order) for each step
    """
    history = []
    for step in range(n_steps):
        print(f"\n--- Optimization step {step+1}/{n_steps} ---")
        # Build circuit
        circuit = build_memory_experiment_circuit(css_code, n_rounds=n_rounds, noise_prob=noise_prob)
        # Analyze errors (with decoding)
        error_rate, confidence_interval, qubit_freqs = analyze_circuit_errors(circuit, n_shots)
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
        print(f"Optimizing CX order for stabilizer {stab_type} {stab_idx}")
        optimize_cx_order(css_code, worst_ancilla[0])
        new_cx_order = css_code.get_cx_order(stab_type, stab_idx)
        history.append((error_rate, confidence_interval, worst_ancilla[0], new_cx_order))
    if not history:
        return None
    return history 