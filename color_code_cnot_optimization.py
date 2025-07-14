import random
from color_code_stim import ColorCode
import matplotlib.pyplot as plt
import numpy as np
import stim
from typing import Tuple, Dict, Union
from collections import defaultdict
import pickle

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
                         n_shots: int = 10000) -> Tuple[float, Tuple[float, float], Dict[int, float]]:
    """
    Analyzes error patterns in a memory experiment circuit and identifies problematic qubits.
    Uses color code decoder to decode syndromes before computing logical error rate.
    
    Args:
        circuit: The memory experiment circuit
        colorcode: ColorCode instance with decode method
        n_shots: Number of shots to sample
        
    Returns:
        Tuple of:
        - Logical error rate (after decoding)
        - Confidence interval (lower, upper) for the error rate
        - Dictionary mapping qubit indices to their error involvement frequency
    """
    # Import the unified function
    from stim_error_analyzer import analyze_circuit_errors_unified
    
    # Use the unified function with color code decoder
    return analyze_circuit_errors_unified(circuit, colorcode.decode, n_shots)


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

def get_initial_cnot_schedule_dict(colorcode, tri_optimal_schedule, init_option='benchmark', history_step=None):
    """Create a dict of lists of lists for the initial schedule.
    
    Args:
        colorcode: The ColorCode instance
        tri_optimal_schedule: The base tri-optimal schedule
        init_option: One of 'benchmark', 'random', 'uniform_random', or 'history'
        history_step: Step number to load from history file (only used if init_option='history')
    """
    tri_optimal_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
    z_target_schedule = np.argsort(tri_optimal_schedule[:6]).tolist()
    x_target_schedule = np.argsort(tri_optimal_schedule[6:]).tolist()

    # To mimic the default, we need to know the number of Z and X stabilizers for d=5
    num_z_stabs = len(colorcode.qubit_groups['anc_Z'])
    num_x_stabs = len(colorcode.qubit_groups['anc_X'])

    if init_option == 'history':
        # Load schedule from history file
        if history_step is None:
            raise ValueError("history_step must be specified when init_option='history'")
        
        try:
            with open('history.pkl', 'rb') as f:
                history = pickle.load(f)
            
            if history_step >= len(history):
                raise ValueError(f"history_step {history_step} is out of range. History has {len(history)} steps (0-indexed)")
            
            # Get the schedule from the specified step
            _, _, _, _, _, cnot_schedule_dict = history[history_step]
            print(f"Loaded CNOT schedule from history step {history_step}")
            return cnot_schedule_dict
            
        except FileNotFoundError:
            print("Warning: history.pkl not found, falling back to benchmark initialization")
            init_option = 'benchmark'
        except Exception as e:
            print(f"Warning: Error loading history step {history_step}: {e}, falling back to benchmark initialization")
            init_option = 'benchmark'
    
    if init_option == 'random':
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
    elif init_option == 'uniform_random':
        # Create one random order for all Z stabilizers and one for all X stabilizers
        z_random_order = z_target_schedule[:]  # Copy the base schedule
        random.shuffle(z_random_order)  # Randomly shuffle it once
        z_schedules = [z_random_order for _ in range(num_z_stabs)]
        
        x_random_order = x_target_schedule[:]  # Copy the base schedule
        random.shuffle(x_random_order)  # Randomly shuffle it once
        x_schedules = [x_random_order for _ in range(num_x_stabs)]
    elif init_option == 'benchmark':
        # Use the same sorted order for all stabilizers (original behavior)
        z_schedules = [z_target_schedule for _ in range(num_z_stabs)]
        x_schedules = [x_target_schedule for _ in range(num_x_stabs)]
    else:
        raise ValueError(f"Invalid init_option: {init_option}. Must be one of 'benchmark', 'random', 'uniform_random', or 'history'")

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

def list_history_steps():
    """List available history steps and their error rates from history.pkl file."""
    try:
        with open('history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        print(f"Available history steps (0-{len(history)-1}):")
        print("Step | Error Rate | Confidence Interval")
        print("-" * 50)
        for i, (error_rate, confidence_interval, worst_ancilla, stab_type, stab_idx, _) in enumerate(history):
            print(f"{i:4d} | {error_rate:.6f} | [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
        return len(history)
    except FileNotFoundError:
        print("No history.pkl file found.")
        return 0
    except Exception as e:
        print(f"Error reading history file: {e}")
        return 0

def main_optimization(
    d=7, rounds=7, n_steps=20, n_shots=100000, p_cnot=1e-3, tri_optimal_schedule=None, 
    init_option='random', history_step=None
):
    if tri_optimal_schedule is None:
        tri_optimal_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]

    # Build initial ColorCode to get stabilizer counts
    colorcode = ColorCode(d=d, rounds=rounds, cnot_schedule="tri_optimal", p_cnot=p_cnot)
    cnot_schedule_dict = get_initial_cnot_schedule_dict(colorcode, tri_optimal_schedule, init_option, history_step)

    history = []
    for step in range(n_steps):
        print(f"\n--- Optimization step {step+1}/{n_steps} ---")
        # Build ColorCode with current schedule
        colorcode = build_color_code_with_schedule(d, rounds, cnot_schedule_dict, p_cnot)
        circuit = colorcode.circuit

        # Analyze errors
        error_rate, confidence_interval, qubit_freqs = analyze_circuit_errors(circuit, colorcode, n_shots)
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
        history.append((error_rate, confidence_interval, worst_ancilla, stab_type, stab_idx, new_cx_order))
    return history, colorcode

if __name__ == "__main__":
    # Example usage with different initialization options:
    
    # Option 1: Benchmark initialization (tri_optimal_schedule)
    # print("Running with benchmark initialization...")
    # history, colorcode = main_optimization(init_option='benchmark')
    
    # Option 2: Random initialization (different random order for each stabilizer)
    print("Running with random initialization...")
    history, colorcode = main_optimization(init_option='random')
    
    # Option 3: Uniform random initialization (same random order for all X stabilizers, same for all Z stabilizers)
    # print("Running with uniform random initialization...")
    # history, colorcode = main_optimization(init_option='uniform_random')
    
    # Option 4: History initialization (load from step 5)
    # print("Running with history initialization from step 5...")
    # history, colorcode = main_optimization(init_option='history', history_step=5)
    
    # Helper: List available history steps
    # list_history_steps()
    
    steps = list(range(1, len(history) + 1))
    logical_error_rates = [r[0] for r in history]
    confidence_intervals = [r[1] for r in history]
    
    # Calculate error bar values
    error_lower = [r - ci[0] for r, ci in zip(logical_error_rates, confidence_intervals)]
    error_upper = [ci[1] - r for r, ci in zip(logical_error_rates, confidence_intervals)]

    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)

    plt.figure(figsize=(8, 5))
    plt.errorbar(steps, logical_error_rates, yerr=[error_lower, error_upper], marker='o', capsize=5, capthick=1)
    plt.xlabel("Optimization Step")
    plt.ylabel("Logical Error Rate")
    plt.title("Logical Error Rate vs Optimization Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print()

    visualize_stabilizer_schedules(colorcode, stabilizer_type='X')
    print()