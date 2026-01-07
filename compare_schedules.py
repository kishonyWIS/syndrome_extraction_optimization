"""
Compare benchmark CNOT schedule vs a custom per-color schedule for color code.

Benchmark schedule: [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2] (uniform for all plaquettes)
Custom schedule: Different schedules for red, green, blue plaquettes

Uses Tesseract decoder from Google Quantum AI:
https://github.com/quantumlib/tesseract-decoder
"""

import numpy as np
import matplotlib.pyplot as plt
from color_code_stim import ColorCode
import stim
from tesseract_decoder import tesseract, utils as tesseract_utils


def schedule_to_cnot_dict(colorcode, schedule):
    """
    Convert a 12-element schedule to a CNOT schedule dict (uniform for all stabilizers).
    
    The schedule is split into two parts:
    - First 6 elements: priorities for Z stabilizers
    - Last 6 elements: priorities for X stabilizers
    
    argsort is used to convert priorities to execution order.
    """
    z_target_schedule = np.argsort(schedule[:6]).tolist()
    x_target_schedule = np.argsort(schedule[6:]).tolist()
    
    num_z_stabs = len(colorcode.qubit_groups['anc_Z'])
    num_x_stabs = len(colorcode.qubit_groups['anc_X'])
    
    return {
        'Z': [z_target_schedule for _ in range(num_z_stabs)],
        'X': [x_target_schedule for _ in range(num_x_stabs)],
    }


def schedule_to_cnot_dict_by_color(colorcode, schedules_by_color):
    """
    Convert per-color 12-element schedules to a per-stabilizer CNOT schedule dict.
    
    Args:
        colorcode: ColorCode instance
        schedules_by_color: Dict like {
            'r': [12-element schedule for red],
            'g': [12-element schedule for green],
            'b': [12-element schedule for blue]
        }
        Each 12-element schedule: first 6 for Z, last 6 for X (as priorities, converted via argsort)
    
    Returns:
        Dictionary with 'Z' and 'X' stabilizer schedules (per-stabilizer lists)
    """
    # Convert each color's schedule to CNOT order
    cnot_orders = {}
    for color, schedule in schedules_by_color.items():
        z_order = np.argsort(schedule[:6]).tolist()
        x_order = np.argsort(schedule[6:]).tolist()
        cnot_orders[color] = {'Z': z_order, 'X': x_order}
    
    z_schedules = []
    x_schedules = []
    
    # Build Z schedules based on each ancilla's color
    for anc_qubit in colorcode.qubit_groups['anc_Z']:
        color = anc_qubit['color']  # 'r', 'g', or 'b'
        z_schedules.append(cnot_orders[color]['Z'])
    
    # Build X schedules based on each ancilla's color
    for anc_qubit in colorcode.qubit_groups['anc_X']:
        color = anc_qubit['color']
        x_schedules.append(cnot_orders[color]['X'])
    
    return {'Z': z_schedules, 'X': x_schedules}


def schedule_to_cnot_dict_by_color_and_type(colorcode, schedules_by_color_and_type):
    """
    Convert per-color and per-type (bulk/edge) schedules to a per-stabilizer CNOT schedule dict.
    
    Bulk plaquettes have 6 data qubits, edge plaquettes have 4 data qubits.
    Edge plaquettes are identified by having a non-None 'boundary' attribute.
    
    Args:
        colorcode: ColorCode instance
        schedules_by_color_and_type: Dict like {
            'r': {'bulk': [12-element schedule], 'edge': [12-element schedule]},
            'g': {'bulk': [12-element schedule], 'edge': [12-element schedule]},
            'b': {'bulk': [12-element schedule], 'edge': [12-element schedule]}
        }
        Each 12-element schedule: first 6 for Z, last 6 for X (as priorities, converted via argsort)
    
    Returns:
        Dictionary with 'Z' and 'X' stabilizer schedules (per-stabilizer lists)
    """
    # Convert each color and type's schedule to CNOT order
    cnot_orders = {}
    for color, type_schedules in schedules_by_color_and_type.items():
        cnot_orders[color] = {}
        for plaq_type, schedule in type_schedules.items():
            z_order = np.argsort(schedule[:6]).tolist()
            x_order = np.argsort(schedule[6:]).tolist()
            cnot_orders[color][plaq_type] = {'Z': z_order, 'X': x_order}
    
    z_schedules = []
    x_schedules = []
    
    # Build Z schedules based on each ancilla's color AND boundary status
    for anc_qubit in colorcode.qubit_groups['anc_Z']:
        color = anc_qubit['color']  # 'r', 'g', or 'b'
        # Determine if bulk or edge based on boundary attribute
        plaq_type = 'edge' if anc_qubit['boundary'] is not None else 'bulk'
        z_schedules.append(cnot_orders[color][plaq_type]['Z'])
    
    # Build X schedules based on each ancilla's color AND boundary status
    for anc_qubit in colorcode.qubit_groups['anc_X']:
        color = anc_qubit['color']
        plaq_type = 'edge' if anc_qubit['boundary'] is not None else 'bulk'
        x_schedules.append(cnot_orders[color][plaq_type]['X'])
    
    return {'Z': z_schedules, 'X': x_schedules}


def create_tesseract_decoder(circuit: stim.Circuit, beam_type: str = 'short'):
    """
    Create a Tesseract decoder for the given circuit.
    
    Args:
        circuit: stim.Circuit to decode
        beam_type: 'short' or 'long' beam configuration
        
    Returns:
        TesseractDecoder instance
    """
    dem = circuit.detector_error_model()
    
    if beam_type == 'long':
        # Long-beam setup (more accurate, slower)
        config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=1_000_000,
            det_beam=20,
            beam_climbing=True,
            det_orders=tesseract_utils.build_det_orders(
                dem=dem,
                num_det_orders=21,
                method=tesseract_utils.DetOrder.DetIndex,
            ),
            no_revisit_dets=True,
        )
    else:
        # Short-beam setup (faster, good balance)
        config = tesseract.TesseractConfig(
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
    
    return tesseract.TesseractDecoder(config)


def analyze_with_tesseract(circuit: stim.Circuit, n_shots: int, beam_type: str = 'short'):
    """
    Analyze circuit errors using Tesseract decoder.
    
    Args:
        circuit: stim.Circuit to analyze
        n_shots: Number of shots to sample
        beam_type: 'short' or 'long' beam configuration
        
    Returns:
        Tuple of (error_rate, confidence_interval, qubit_freqs)
    """
    # Create sampler and decoder
    sampler = circuit.compile_detector_sampler()
    decoder = create_tesseract_decoder(circuit, beam_type)
    
    # Sample detection events and observables
    detection_events, observable_flips = sampler.sample(
        shots=n_shots, 
        separate_observables=True
    )
    
    # Decode each shot
    num_errors = 0
    for shot_idx in range(n_shots):
        # Get detection events for this shot
        dets = detection_events[shot_idx]
        
        # Decode
        predicted_obs = decoder.decode(dets)
        
        # Check if prediction matches actual observable
        actual_obs = observable_flips[shot_idx]
        if not np.array_equal(predicted_obs, actual_obs):
            num_errors += 1
    
    error_rate = num_errors / n_shots
    
    # Wilson score confidence interval
    from scipy import stats
    if n_shots > 0:
        ci = stats.binomtest(num_errors, n_shots).proportion_ci(confidence_level=0.95)
        confidence_interval = (ci.low, ci.high)
    else:
        confidence_interval = (0.0, 1.0)
    
    print(f"Logical error rate (Tesseract {beam_type}-beam): {error_rate:.6f} [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
    
    return error_rate, confidence_interval, {}


def analyze_schedule(name, schedule, d=7, rounds=2, p_cnot=1e-3, n_shots=10000, 
                    is_per_color=False, decoder_type='tesseract', beam_type='short'):
    """
    Analyze a schedule and return metrics.
    
    Args:
        name: Name for display
        schedule: Either a 12-element list (uniform) or dict with 'r', 'g', 'b' keys (per-color)
        is_per_color: If True, schedule is a per-color dict
        decoder_type: 'tesseract' or 'colorcode' (built-in)
        beam_type: For tesseract decoder: 'short' or 'long'
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Decoder: {decoder_type}" + (f" ({beam_type}-beam)" if decoder_type == 'tesseract' else ""))
    if is_per_color:
        print(f"Per-color schedules:")
        for color, sched in schedule.items():
            print(f"  {color}: {sched}")
    else:
        print(f"Schedule: {schedule}")
    print(f"{'='*60}")
    
    # Build initial code to get stabilizer counts
    colorcode_init = ColorCode(d=d, rounds=rounds, cnot_schedule="tri_optimal", p_cnot=p_cnot)
    
    # Convert schedule to dict format
    if is_per_color:
        cnot_schedule_dict = schedule_to_cnot_dict_by_color(colorcode_init, schedule)
        # Show CNOT orders for each color
        for color in ['r', 'g', 'b']:
            z_order = np.argsort(schedule[color][:6]).tolist()
            x_order = np.argsort(schedule[color][6:]).tolist()
            print(f"{color.upper()} plaquettes - Z order: {z_order}, X order: {x_order}")
    else:
        cnot_schedule_dict = schedule_to_cnot_dict(colorcode_init, schedule)
        print(f"Z stabilizer CNOT order: {cnot_schedule_dict['Z'][0]}")
        print(f"X stabilizer CNOT order: {cnot_schedule_dict['X'][0]}")
    
    # Build the color code with this schedule
    colorcode = ColorCode(
        d=d,
        rounds=rounds,
        cnot_schedule=cnot_schedule_dict,
        p_cnot=p_cnot,
    )
    circuit = colorcode.circuit
    
    # Calculate circuit-level distance
    undetectable_errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False
    )
    circuit_distance = len(undetectable_errors)
    print(f"Circuit-level distance: {circuit_distance}")
    
    # Calculate graphlike distance
    graphlike_distance = len(circuit.shortest_graphlike_error())
    print(f"Graphlike distance: {graphlike_distance}")
    
    # Analyze error rate with chosen decoder
    if decoder_type == 'tesseract':
        error_rate, confidence_interval, qubit_freqs = analyze_with_tesseract(
            circuit, n_shots, beam_type
        )
    else:
        # Use built-in color code decoder
        from stim_error_analyzer import analyze_circuit_errors_unified
        error_rate, confidence_interval, qubit_freqs = analyze_circuit_errors_unified(
            circuit, colorcode.decode, n_shots
        )
        print(f"Logical error rate (ColorCode decoder): {error_rate:.6f} [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
    
    return {
        'name': name,
        'schedule': schedule,
        'circuit_distance': circuit_distance,
        'graphlike_distance': graphlike_distance,
        'error_rate': error_rate,
        'confidence_interval': confidence_interval,
        'qubit_freqs': qubit_freqs,
    }


def compare_schedules(d=7, rounds=2, p_cnot=1e-3, n_shots=100000, 
                     decoder_type='tesseract', beam_type='short'):
    """
    Compare benchmark and custom per-color schedules.
    
    Args:
        decoder_type: 'tesseract' or 'colorcode'
        beam_type: For tesseract: 'short' or 'long'
    """
    # Benchmark schedule (uniform for all plaquettes)
    benchmark_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
    
    # Custom per-color schedules
    # Red:   [1,4,2,6,3,5] + [1+6,4+6,2+6,6+6,3+6,5+6] = [1,4,2,6,3,5,7,10,8,12,9,11]
    # Blue:  [1,5,3,6,2,4] + [7,11,9,12,8,10]
    # Green: [4,2,6,3,5,1] + [10,8,12,9,11,7]
    custom_schedules_by_color = {
        'r': [1, 4, 2, 6, 3, 5],
        'b': [1, 5, 3, 6, 2, 4],
        'g': [4, 2, 6, 3, 5, 1],
    }
    for color in ['r', 'g', 'b']:
        custom_schedules_by_color[color] = custom_schedules_by_color[color] + [x + 6 for x in custom_schedules_by_color[color]]
    
    # Analyze both
    benchmark_results = analyze_schedule(
        "Benchmark (tri-optimal, uniform)", 
        benchmark_schedule, 
        d=d, rounds=rounds, p_cnot=p_cnot, n_shots=n_shots,
        is_per_color=False,
        decoder_type=decoder_type,
        beam_type=beam_type
    )
    
    custom_results = analyze_schedule(
        "Custom (per-color)", 
        custom_schedules_by_color, 
        d=d, rounds=rounds, p_cnot=p_cnot, n_shots=n_shots,
        is_per_color=True,
        decoder_type=decoder_type,
        beam_type=beam_type
    )
    
    # Print comparison summary
    print("\n" + "="*60)
    print(f"COMPARISON SUMMARY (Decoder: {decoder_type})")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Benchmark':<20} {'Custom (per-color)':<20}")
    print("-" * 65)
    print(f"{'Circuit distance':<25} {benchmark_results['circuit_distance']:<20} {custom_results['circuit_distance']:<20}")
    print(f"{'Graphlike distance':<25} {benchmark_results['graphlike_distance']:<20} {custom_results['graphlike_distance']:<20}")
    print(f"{'Error rate':<25} {benchmark_results['error_rate']:<20.6f} {custom_results['error_rate']:<20.6f}")
    
    # Determine winner
    if benchmark_results['error_rate'] > 0:
        error_diff = custom_results['error_rate'] - benchmark_results['error_rate']
        if error_diff < 0:
            improvement = -error_diff / benchmark_results['error_rate'] * 100
            print(f"\n✓ Custom schedule is BETTER by {improvement:.2f}%")
        elif error_diff > 0:
            degradation = error_diff / benchmark_results['error_rate'] * 100
            print(f"\n✗ Custom schedule is WORSE by {degradation:.2f}%")
        else:
            print("\n= Schedules are equivalent")
    else:
        if custom_results['error_rate'] == 0:
            print("\n= Both schedules have zero errors")
        else:
            print(f"\n✗ Custom has errors while benchmark has none")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of error rates
    ax1 = axes[0]
    names = ['Benchmark', 'Custom\n(per-color)']
    error_rates = [benchmark_results['error_rate'], custom_results['error_rate']]
    ci_lower = [r['error_rate'] - r['confidence_interval'][0] for r in [benchmark_results, custom_results]]
    ci_upper = [r['confidence_interval'][1] - r['error_rate'] for r in [benchmark_results, custom_results]]
    
    bars = ax1.bar(names, error_rates, yerr=[ci_lower, ci_upper], capsize=10, color=['#2196F3', '#4CAF50'])
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_title(f'Error Rate Comparison ({decoder_type} decoder)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, error_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{rate:.6f}', ha='center', va='bottom', fontsize=10)
    
    # Bar chart of distances
    ax2 = axes[1]
    x = np.arange(2)
    width = 0.35
    
    circuit_dists = [benchmark_results['circuit_distance'], custom_results['circuit_distance']]
    graphlike_dists = [benchmark_results['graphlike_distance'], custom_results['graphlike_distance']]
    
    bars1 = ax2.bar(x - width/2, circuit_dists, width, label='Circuit Distance', color='#FF9800')
    bars2 = ax2.bar(x + width/2, graphlike_dists, width, label='Graphlike Distance', color='#9C27B0')
    
    ax2.set_ylabel('Distance')
    ax2.set_title('Code Distance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('schedule_comparison.png', dpi=150)
    plt.show()
    
    return benchmark_results, custom_results


if __name__ == "__main__":
    # Run comparison with Tesseract decoder
    # Use 'short' beam for faster results, 'long' for higher accuracy
    benchmark_results, custom_results = compare_schedules(
        d=7,
        rounds=2,
        p_cnot=3e-3,
        n_shots=100,  # Reduced from 10M since Tesseract is slower per-shot
        decoder_type='tesseract',
        beam_type='short'
    )
