import random
from color_code_stim import ColorCode
import matplotlib.pyplot as plt
import numpy as np
import stim
from typing import Tuple, Dict, Union
from collections import defaultdict

def draw_lattice_with_individual_colors(code, qubit_colors=None, **kwargs):
    """
    Draw the lattice with individual qubit colors.
    
    Args:
        code: ColorCode instance
        qubit_colors: Dictionary mapping qubit names (str) to colors (str)
        **kwargs: Additional arguments passed to draw_lattice
    
    Returns:
        matplotlib.axes.Axes
    """
    if qubit_colors is None:
        return code.draw_lattice(**kwargs)
    
    # Get the base lattice
    ax = code.draw_lattice(show_data_qubits=False, **kwargs)  # Don't draw data qubits yet
    
    # Get data qubits from the tanner graph
    graph = code.tanner_graph
    data_qubits = graph.vs.select(pauli=None)
    
    # Draw each qubit individually with its assigned color
    for qubit in data_qubits:
        qubit_name = qubit["name"]
        x, y = qubit["x"], qubit["y"]
        
        # Get color for this qubit, default to black if not specified
        color = qubit_colors.get(qubit_name, "black")
        
        # Draw the qubit
        ax.scatter(
            [x], [y],
            c=color,
            s=kwargs.get('data_qubit_size', 100.0),
            edgecolors="black",
            linewidths=1,
            marker="o",
            zorder=2,
        )
    
    return ax

def analyze_circuit_errors(circuit: stim.Circuit, colorcode: ColorCode,
                         n_shots: int = 10000) -> Tuple[float, Dict[int, float]]:
    """
    Analyzes error patterns in a memory experiment circuit and identifies problematic qubits.
    Uses color code decoder to decode syndromes before computing logical error rate.
    
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
    
    # List out the errors, for later lookup
    error_instructions = []
    for instruction in dem.flattened():
        if instruction.type == 'error':
            error_instructions.append(instruction)
    
    sampler = dem.compile_sampler()
    
    # Sample with error tracking
    sample_results: Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]] = sampler.sample(shots=n_shots, return_errors=True)
    dets, obs, errs = sample_results
    
    if errs is None:
        print("No error information returned from sampler")
        return 0.0, {}
        
    # Decode each shot
    predicted_observables = colorcode.decode(dets)
    
    # Handle the return type - decode returns either ndarray or Tuple[ndarray, dict]
    if isinstance(predicted_observables, tuple):
        predicted_observables = predicted_observables[0]  # Extract the array from the tuple
    
    # Ensure predicted_observables has the right shape for comparison
    if predicted_observables.ndim == 1:
        predicted_observables = predicted_observables.reshape(-1, 1)
    
    # Check if any logical operator was wrongly corrected
    # For multiple observables, check if any observable was wrongly corrected
    logical_errors = np.logical_xor(predicted_observables, obs).any(axis=1)  # Any observable wrong
    logical_error_rate = float(np.mean(logical_errors))
    print(f"Logical error rate (after decoding): {logical_error_rate}")
    
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
    
    return logical_error_rate, qubit_frequencies


def randomize_stabilizer_cx_order(cnot_schedule_dict, stab_type, stab_idx):
    """Randomly permute the CX order for a given stabilizer in the cnot_schedule dict."""
    order = cnot_schedule_dict[stab_type][stab_idx]
    new_order = order[:]
    random.shuffle(new_order)
    cnot_schedule_dict[stab_type][stab_idx] = new_order

def build_color_code_with_schedule(d, rounds, cnot_schedule_dict, p_cnot=1e-3):
    """Build a ColorCode with the given per-stabilizer cnot_schedule dict."""
    return ColorCode(
        d=d,
        rounds=rounds,
        cnot_schedule=cnot_schedule_dict,
        p_cnot=p_cnot,
    )

def get_initial_cnot_schedule_dict(colorcode, tri_optimal_schedule, random_init=False):
    """Create a dict of lists of lists for the initial schedule.
    
    Args:
        colorcode: The ColorCode instance
        tri_optimal_schedule: The base tri-optimal schedule
        random_init: If True, each stabilizer gets a random order. If False, all stabilizers use the same sorted order.
    """

    tri_optimal_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
    z_target_schedule = np.argsort(tri_optimal_schedule[:6]).tolist()
    x_target_schedule = np.argsort(tri_optimal_schedule[6:]).tolist()

    # To mimic the default, we need to know the number of Z and X stabilizers for d=5
    num_z_stabs = len(colorcode.qubit_groups['anc_Z'])
    num_x_stabs = len(colorcode.qubit_groups['anc_X'])

    if random_init:
        # Create random orders for each stabilizer
        z_schedules = []
        for _ in range(num_z_stabs):
            random_order = z_target_schedule[:]  # Copy the base schedule
            random.shuffle(random_order)  # Randomly shuffle it
            z_schedules.append(random_order)
        
        x_schedules = []
        for _ in range(num_x_stabs):
            random_order = x_target_schedule[:]  # Copy the base schedule
            random.shuffle(random_order)  # Randomly shuffle it
            x_schedules.append(random_order)
    else:
        # Use the same sorted order for all stabilizers (original behavior)
        z_schedules = [z_target_schedule for _ in range(num_z_stabs)]
        x_schedules = [x_target_schedule for _ in range(num_x_stabs)]

    cnot_schedule_dict = {
        'Z': z_schedules,
        'X': x_schedules,
    }

    return cnot_schedule_dict

def visualize_stabilizer_schedules(colorcode, stabilizer_type='X', colormap_name='autumn'):
    """
    Visualize the CNOT schedules for stabilizers using a colormap to show the order.
    
    Args:
        colorcode: ColorCode instance
        stabilizer_type: 'X' or 'Z' stabilizers to visualize
        colormap_name: Name of matplotlib colormap to use
    """
    anc_qubits = colorcode.qubit_groups[f'anc_{stabilizer_type}']
    schedules = colorcode.cnot_schedule[stabilizer_type]
    tanner_graph = colorcode.tanner_graph
    
    # Define offsets for X and Z stabilizers
    if stabilizer_type == 'X':
        offsets = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
    else:  # Z stabilizers
        offsets = [(-1, 2), (1, 2), (2, 0), (1, -2), (-1, -2), (-2, 0)]

    for stab_idx, anc_qubit in enumerate(anc_qubits):
        schedule = schedules[stab_idx]
        data_qubits_list = []
        
        for offset_idx in schedule:
            offset = offsets[offset_idx % 6]
            data_qubit_x = anc_qubit["x"] + offset[0]
            data_qubit_y = anc_qubit["y"] + offset[1]
            data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
            
            try:
                data_qubit = tanner_graph.vs.find(name=data_qubit_name)
            except ValueError:
                continue
            data_qubits_list.append(data_qubit_name)
        
        print(f"Stabilizer {stab_idx} data qubits: {data_qubits_list}")
        
        # Create colormap for the qubits
        if data_qubits_list:
            # Create colors for each qubit based on its position in the list
            qubit_colors = {}
            
            # Use matplotlib's built-in colormap
            import matplotlib.pyplot as plt
            
            for i, qubit_name in enumerate(data_qubits_list):
                if len(data_qubits_list) > 1:
                    # Normalize position to [0, 1] for colormap
                    color_value = i / (len(data_qubits_list) - 1)
                else:
                    color_value = 0.5  # Middle of colormap for single qubit
                
                color = plt.get_cmap(colormap_name)(color_value)
                qubit_colors[qubit_name] = color
            
            # Draw with individual colors using colormap
            draw_lattice_with_individual_colors(colorcode, qubit_colors=qubit_colors)
            plt.title(f"Stabilizer {stab_idx} CNOT Schedule (Order: {schedule})")
            plt.show()
        else:
            print(f"No valid data qubits found for stabilizer {stab_idx}")

def main_optimization(
    d=7, rounds=7, n_steps=30, n_shots=100000, p_cnot=5e-3, tri_optimal_schedule=None, random_init=True
):
    if tri_optimal_schedule is None:
        tri_optimal_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]

    # Build initial ColorCode to get stabilizer counts
    colorcode = ColorCode(d=d, rounds=rounds, cnot_schedule="tri_optimal", p_cnot=p_cnot)
    cnot_schedule_dict = get_initial_cnot_schedule_dict(colorcode, tri_optimal_schedule, random_init)

    history = []
    for step in range(n_steps):
        print(f"\n--- Optimization step {step+1}/{n_steps} ---")
        # Build ColorCode with current schedule
        colorcode = build_color_code_with_schedule(d, rounds, cnot_schedule_dict, p_cnot)
        circuit = colorcode.circuit
        # Analyze errors
        error_rate, qubit_freqs = analyze_circuit_errors(circuit, colorcode, n_shots)
        if not qubit_freqs:
            print("No problematic qubits found")
            break
        # Find most problematic ancilla among the qubits
        ancilla_freqs = {q: freq for q, freq in qubit_freqs.items()
                         if q in colorcode.qubit_groups['anc_Z']['qid'] or q in colorcode.qubit_groups['anc_X']['qid']}
        if not ancilla_freqs:
            print("No ancilla qubits involved in errors. Stopping optimization.")
            break
        worst_ancilla = max(ancilla_freqs.items(), key=lambda x: x[1])[0]
        print(f"Most problematic ancilla: {worst_ancilla}")
        # Determine if it's an X or Z stabilizer ancilla and optimize its CX order
        if worst_ancilla in colorcode.qubit_groups['anc_Z']['qid']:
            stab_type = 'Z'
            stab_idx = list(colorcode.qubit_groups['anc_Z']['qid']).index(worst_ancilla)
        else:
            stab_type = 'X'
            stab_idx = list(colorcode.qubit_groups['anc_X']['qid']).index(worst_ancilla)
        print(f"Optimizing CX order for stabilizer {stab_type} {stab_idx}")
        randomize_stabilizer_cx_order(cnot_schedule_dict, stab_type, stab_idx)
        new_cx_order = cnot_schedule_dict.copy()
        history.append((error_rate, worst_ancilla, stab_type, stab_idx, new_cx_order))
    return history, colorcode

if __name__ == "__main__":
    history, colorcode = main_optimization() 
    steps = list(range(1, len(history) + 1))
    logical_error_rates = [r[0] for r in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, logical_error_rates, marker='o')
    plt.xlabel("Optimization Step")
    plt.ylabel("Logical Error Rate")
    plt.title("Logical Error Rate vs Optimization Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print()

    # Visualize X stabilizer CNOT schedules
    visualize_stabilizer_schedules(colorcode, stabilizer_type='X')
    print()