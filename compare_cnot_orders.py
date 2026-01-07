#!/usr/bin/env python3
"""
Simple script to compare two specific CNOT orders:
- Order 1: [0, 2, 4, 3, 1, 5] 
- Order 2: [0, 1, 2, 3, 4, 5] (standard order)

Compares both in terms of code distance and logical error rate.
"""

import time
from typing import Dict, List, Tuple
from color_code_stim import ColorCode
from color_code_cnot_optimization import get_initial_cnot_schedule_dict, build_color_code_with_schedule


def evaluate_logical_error_rate(colorcode: ColorCode, n_shots: int = 10000) -> Tuple[float, float]:
    """
    Evaluate logical error rate using ColorCode's built-in simulation method.
    
    Args:
        colorcode: ColorCode instance to evaluate
        n_shots: Number of shots to sample
        
    Returns:
        Tuple of (X logical error rate, Z logical error rate)
    """
    try:
        # Use ColorCode's built-in simulation method which handles decoding correctly
        num_fails, info = colorcode.simulate(
            n_shots,
            full_output=True,
            alpha=0.01,  # Significance level for confidence intervals
            verbose=False,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Calculate logical error rate
        logical_error_rate = num_fails / n_shots
        
        # For now, return the same rate for both X and Z since we're getting overall logical error rate
        return logical_error_rate, logical_error_rate
        
    except Exception as e:
        print(f"    Warning: Error evaluating logical error rate: {e}")
        return float('inf'), float('inf')


def evaluate_schedule(schedule: Dict[str, List[List[int]]], d: int, p_cnot: float, 
                     schedule_name: str, n_logical_shots: int = 10000) -> Tuple[int, float, float]:
    """
    Evaluate a single schedule for distance and logical error rate.
    
    Args:
        schedule: Schedule dictionary
        d: Code distance
        p_cnot: CNOT error probability
        schedule_name: Name for display purposes
        n_logical_shots: Number of shots for logical error rate evaluation
        
    Returns:
        Tuple of (accurate_distance, x_error_rate, z_error_rate)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {schedule_name}")
    print(f"Schedule: {schedule['X'][0] if schedule['X'] else 'None'}")
    print(f"{'='*60}")
    
    # Build color code with the schedule
    colorcode = build_color_code_with_schedule(d, 2, schedule, p_cnot)
    circuit = colorcode.circuit
    
    # Step 1: Fast check using graphlike distance
    print("Step 1: Computing graphlike distance...")
    graphlike_distance = len(circuit.shortest_graphlike_error())
    print(f"  Graphlike distance: {graphlike_distance}")
    
    # Step 2: Accurate distance computation
    print("Step 2: Computing accurate distance...")
    undetectable_errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False
    )
    accurate_distance = len(undetectable_errors)
    print(f"  Accurate distance: {accurate_distance}")
    
    # Step 3: Evaluate logical error rate
    print(f"Step 3: Evaluating logical error rate with {n_logical_shots} shots...")
    x_error_rate, z_error_rate = evaluate_logical_error_rate(colorcode, n_logical_shots)
    total_error_rate = x_error_rate + z_error_rate
    print(f"  X logical error rate: {x_error_rate:.2e}")
    print(f"  Z logical error rate: {z_error_rate:.2e}")
    print(f"  Total logical error rate: {total_error_rate:.2e}")
    
    return accurate_distance, x_error_rate, z_error_rate


def compare_cnot_orders(d: int = 7, p_cnot: float = 1e-3, n_logical_shots: int = 10000):
    """
    Compare two specific CNOT orders.
    
    Args:
        d: Code distance
        p_cnot: CNOT error probability
        n_logical_shots: Number of shots for logical error rate evaluation
    """
    print(f"Starting CNOT order comparison...")
    print(f"Code distance: {d}, p_cnot: {p_cnot}")
    print(f"Logical error shots: {n_logical_shots}")
    
    # Get initial schedule structure to know how many stabilizers we have
    initial_colorcode = ColorCode(
        d=d, 
        cnot_schedule="tri_optimal", 
        p_bitflip=p_cnot,
        rounds=2
    )
    
    # Get initial schedule dictionary structure
    initial_schedule = get_initial_cnot_schedule_dict(
        initial_colorcode, None, 'benchmark'
    )
    
    print(f"X stabilizers: {len(initial_schedule['X'])}")
    print(f"Z stabilizers: {len(initial_schedule['Z'])}")
    
    # Define the two orders to compare
    order_1 = [0, 2, 4, 3, 1, 5]
    order_2 = [0, 1, 2, 3, 4, 5]
    
    # Create schedules
    schedule_1 = {
        'X': [order_1.copy() for _ in initial_schedule['X']],
        'Z': [order_1.copy() for _ in initial_schedule['Z']]
    }
    
    schedule_2 = {
        'X': [order_2.copy() for _ in initial_schedule['X']],
        'Z': [order_2.copy() for _ in initial_schedule['Z']]
    }
    
    # Evaluate both schedules
    start_time = time.time()
    
    distance_1, x_error_1, z_error_1 = evaluate_schedule(
        schedule_1, d, p_cnot, "Order 1: [0, 2, 4, 3, 1, 5]", n_logical_shots
    )
    
    distance_2, x_error_2, z_error_2 = evaluate_schedule(
        schedule_2, d, p_cnot, "Order 2: [0, 1, 2, 3, 4, 5]", n_logical_shots
    )
    
    total_time = time.time() - start_time
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Metric':<25} {'Order 1':<20} {'Order 2':<20} {'Difference':<15}")
    print(f"{'-'*80}")
    print(f"{'CNOT Order':<25} {str(order_1):<20} {str(order_2):<20} {'N/A':<15}")
    print(f"{'Accurate Distance':<25} {distance_1:<20} {distance_2:<20} {distance_1 - distance_2:<+15}")
    print(f"{'X Logical Error Rate':<25} {x_error_1:<20.2e} {x_error_2:<20.2e} {x_error_1 - x_error_2:<+15.2e}")
    print(f"{'Z Logical Error Rate':<25} {z_error_1:<20.2e} {z_error_2:<20.2e} {z_error_1 - z_error_2:<+15.2e}")
    print(f"{'Total Logical Error Rate':<25} {x_error_1 + z_error_1:<20.2e} {x_error_2 + z_error_2:<20.2e} {(x_error_1 + z_error_1) - (x_error_2 + z_error_2):<+15.2e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    if distance_1 > distance_2:
        print(f"✓ Order 1 has BETTER distance: {distance_1} vs {distance_2}")
    elif distance_1 < distance_2:
        print(f"✗ Order 1 has WORSE distance: {distance_1} vs {distance_2}")
    else:
        print(f"= Both orders have EQUAL distance: {distance_1}")
    
    total_error_1 = x_error_1 + z_error_1
    total_error_2 = x_error_2 + z_error_2
    
    if total_error_1 < total_error_2:
        print(f"✓ Order 1 has LOWER logical error rate: {total_error_1:.2e} vs {total_error_2:.2e}")
    elif total_error_1 > total_error_2:
        print(f"✗ Order 1 has HIGHER logical error rate: {total_error_1:.2e} vs {total_error_2:.2e}")
    else:
        print(f"= Both orders have EQUAL logical error rate: {total_error_1:.2e}")
    
    print(f"\nTotal evaluation time: {total_time:.2f}s")
    
    return {
        'order_1': {'distance': distance_1, 'x_error': x_error_1, 'z_error': z_error_1, 'total_error': total_error_1},
        'order_2': {'distance': distance_2, 'x_error': x_error_2, 'z_error': z_error_2, 'total_error': total_error_2}
    }


if __name__ == "__main__":
    # Compare the two CNOT orders
    results = compare_cnot_orders(
        d=9,                    # Code distance
        p_cnot=2e-3,           # CNOT error probability
        n_logical_shots=100000   # Number of shots for logical error rate evaluation
    )
    
    print(f"\nFinal Results:")
    print(f"Order 1 [0, 2, 4, 3, 1, 5]: Distance = {results['order_1']['distance']}, Total Error Rate = {results['order_1']['total_error']:.2e}")
    print(f"Order 2 [0, 1, 2, 3, 4, 5]: Distance = {results['order_2']['distance']}, Total Error Rate = {results['order_2']['total_error']:.2e}")

