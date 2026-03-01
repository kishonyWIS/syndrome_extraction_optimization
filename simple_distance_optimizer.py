#!/usr/bin/env python3
"""
Simple CNOT schedule optimizer to maximize code distance.
Samples random schedules and uses two-tier distance calculation for efficiency.
Also evaluates logical error rates using Stim sampling and ColorCode decoding.
"""

import random
import time
import itertools
import numpy as np
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
        # In a more sophisticated implementation, we could separate X and Z errors
        return logical_error_rate, logical_error_rate
        
    except Exception as e:
        print(f"    Warning: Error evaluating logical error rate: {e}")
        return float('inf'), float('inf')


def simple_distance_optimization(d: int, n_samples: int = 100, p_cnot: float = 1e-3, 
                               use_row_based_permutations: bool = False, exhaustive_search: bool = False,
                               evaluate_logical_errors: bool = True, n_logical_shots: int = 10000):
    """
    Simple optimization: sample random CNOT schedules to maximize code distance.
    
    Args:
        d: Code distance
        n_samples: Number of random schedules to try (ignored if exhaustive_search=True)
        p_cnot: CNOT error probability (passed as p_bitflip to ColorCode)
        use_row_based_permutations: If True, use different permutations for even vs. odd rows
        exhaustive_search: If True, try all possible schedules starting with 0
        evaluate_logical_errors: If True, also evaluate logical error rates
        n_logical_shots: Number of shots for logical error rate evaluation
    """
    
    print(f"Starting simple distance optimization...")
    print(f"Code distance: {d}, p_cnot: {p_cnot}")
    if exhaustive_search:
        print(f"Exhaustive search: All permutations starting with 0")
        n_samples = 120  # 5! = 120 permutations
    else:
        print(f"Samples: {n_samples}")
    print(f"Row-based permutations: {use_row_based_permutations}")
    print(f"Evaluate logical errors: {evaluate_logical_errors}")
    if evaluate_logical_errors:
        print(f"Logical error shots: {n_logical_shots}")
    print("-" * 50)
    
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
    
    # Track best results
    best_distance = 0
    best_schedule = None
    best_logical_error_rate = float('inf')
    best_x_error_rate = float('inf')
    best_z_error_rate = float('inf')
    fast_checks = 0
    accurate_checks = 0
    
    start_time = time.time()
    
    if exhaustive_search:
        # Generate all permutations starting with 0
        all_permutations = []
        for perm in itertools.permutations([1, 2, 3, 4, 5]):
            all_permutations.append([0] + list(perm))
        
        print(f"Generated {len(all_permutations)} unique schedules to test")
        
        for sample_idx, permutation in enumerate(all_permutations):
            print(f"\nSchedule {sample_idx + 1}/{len(all_permutations)}: {permutation}")
            
            # Apply same permutation to all X and Z stabilizers
            schedule = {
                'X': [permutation.copy() for _ in initial_schedule['X']],
                'Z': [permutation.copy() for _ in initial_schedule['Z']]
            }
            
            # Step 1: Fast check using graphlike distance
            colorcode = build_color_code_with_schedule(d, 2, schedule, p_cnot)
            circuit = colorcode.circuit
            
            graphlike_distance = len(circuit.shortest_graphlike_error())
            print(f"  Graphlike distance: {graphlike_distance}")
            
            # If graphlike distance is lower than best, skip accurate computation
            if graphlike_distance <= best_distance:
                print(f"  Skipping accurate computation (graphlike ≤ best)")
                fast_checks += 1
                continue
                    
            # Step 2: Accurate distance computation
            undetectable_errors = circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=9999,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False,
                canonicalize_circuit_errors=False
            )
            # undetectable_errors = circuit.search_for_undetectable_logical_errors(
            #     dont_explore_detection_event_sets_with_size_above=8,
            #     dont_explore_edges_with_degree_above=8,
            #     dont_explore_edges_increasing_symptom_degree=True,
            # )
            accurate_distance = len(undetectable_errors)
            print(f"  Accurate distance: {accurate_distance}")
            
            accurate_checks += 1
            
            # Step 3: Evaluate logical error rate if requested
            if evaluate_logical_errors:
                print(f"  Evaluating logical error rate with {n_logical_shots} shots...")
                x_error_rate, z_error_rate = evaluate_logical_error_rate(colorcode, n_logical_shots)
                total_error_rate = x_error_rate + z_error_rate
                print(f"  X logical error rate: {x_error_rate:.2e}")
                print(f"  Z logical error rate: {z_error_rate:.2e}")
                print(f"  Total logical error rate: {total_error_rate:.2e}")
            else:
                x_error_rate = z_error_rate = total_error_rate = float('inf')
            
            # Update best if we found improvement in distance OR logical error rate
            improved = False
            if accurate_distance > best_distance:
                print(f"  NEW BEST DISTANCE! Distance: {accurate_distance} (previous: {best_distance})")
                improved = True
            elif accurate_distance == best_distance and total_error_rate < best_logical_error_rate:
                print(f"  NEW BEST LOGICAL ERROR RATE! Rate: {total_error_rate:.2e} (previous: {best_logical_error_rate:.2e})")
                improved = True
            
            if improved:
                best_distance = accurate_distance
                best_schedule = schedule.copy()
                best_logical_error_rate = total_error_rate
                best_x_error_rate = x_error_rate
                best_z_error_rate = z_error_rate
            else:
                print(f"  No improvement (current best distance: {best_distance}, error rate: {best_logical_error_rate:.2e})")
    else:
        # Original random sampling logic
        for sample_idx in range(n_samples):
            print(f"\nSample {sample_idx + 1}/{n_samples}")
            
            if use_row_based_permutations:
                # Generate different random orders for even vs. odd rows
                schedule = generate_row_based_schedule(initial_colorcode, initial_schedule)
                print(f"  Row-based schedule generated")
            else:
                # Generate random schedule: same random order for all plaquettes
                random_order = list(range(6))  # [0, 1, 2, 3, 4, 5]
                random.shuffle(random_order)
                
                # Apply same random order to all X and Z stabilizers
                schedule = {
                    'X': [random_order.copy() for _ in initial_schedule['X']],
                    'Z': [random_order.copy() for _ in initial_schedule['Z']]
                }
                print(f"  Random order: {random_order}")
            
            # Step 1: Fast check using graphlike distance
            colorcode = build_color_code_with_schedule(d, 2, schedule, p_cnot)
            circuit = colorcode.circuit
            
            graphlike_distance = len(circuit.shortest_graphlike_error())
            print(f"  Graphlike distance: {graphlike_distance}")
            
            # If graphlike distance is lower than best, skip accurate computation
            if graphlike_distance <= best_distance:
                print(f"  Skipping accurate computation (graphlike ≤ best)")
                fast_checks += 1
                continue
                    
            # Step 2: Accurate distance computation
            undetectable_errors = circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=9999,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False,
                canonicalize_circuit_errors=False
            )
            accurate_distance = len(undetectable_errors)
            print(f"  Accurate distance: {accurate_distance}")
            
            accurate_checks += 1
            
            # Step 3: Evaluate logical error rate if requested
            if evaluate_logical_errors:
                print(f"  Evaluating logical error rate with {n_logical_shots} shots...")
                x_error_rate, z_error_rate = evaluate_logical_error_rate(colorcode, n_logical_shots)
                total_error_rate = x_error_rate + z_error_rate
                print(f"  X logical error rate: {x_error_rate:.2e}")
                print(f"  Z logical error rate: {z_error_rate:.2e}")
                print(f"  Total logical error rate: {total_error_rate:.2e}")
            else:
                x_error_rate = z_error_rate = total_error_rate = float('inf')
            
            # Update best if we found improvement in distance OR logical error rate
            improved = False
            if accurate_distance > best_distance:
                print(f"  NEW BEST DISTANCE! Distance: {accurate_distance} (previous: {best_distance})")
                improved = True
            elif accurate_distance == best_distance and total_error_rate < best_logical_error_rate:
                print(f"  NEW BEST LOGICAL ERROR RATE! Rate: {total_error_rate:.2e} (previous: {best_logical_error_rate:.2e})")
                improved = True
            
            if improved:
                best_distance = accurate_distance
                best_schedule = schedule.copy()
                best_logical_error_rate = total_error_rate
                best_x_error_rate = x_error_rate
                best_z_error_rate = z_error_rate
            else:
                print(f"  No improvement (current best distance: {best_distance}, error rate: {best_logical_error_rate:.2e})")
                    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    if exhaustive_search:
        print(f"Total schedules tested: {len(all_permutations) if exhaustive_search else n_samples}")
    else:
        print(f"Total samples: {n_samples}")
    print(f"Best distance found: {best_distance}")
    if evaluate_logical_errors:
        print(f"Best logical error rates:")
        print(f"  X logical: {best_x_error_rate:.2e}")
        print(f"  Z logical: {best_z_error_rate:.2e}")
        print(f"  Total logical: {best_logical_error_rate:.2e}")
    print(f"Fast checks (skipped): {fast_checks}")
    print(f"Accurate checks: {accurate_checks}")
    print(f"Total time: {total_time:.2f}s")
    if exhaustive_search:
        print(f"Average time per schedule: {total_time/len(all_permutations):.3f}s")
    else:
        print(f"Average time per sample: {total_time/n_samples:.3f}s")
    
    if best_schedule:
        print(f"\nBest schedule:")
        print(f"  X stabilizers: {len(best_schedule['X'])}")
        print(f"  Z stabilizers: {len(best_schedule['Z'])}")
        if not use_row_based_permutations and best_schedule['X']:
            print(f"  Order: {best_schedule['X'][0] if best_schedule['X'] else 'None'}")
        else:
            print(f"  Row-based permutations used")
    
    return best_distance, best_schedule, best_logical_error_rate


def generate_row_based_schedule(colorcode: ColorCode, initial_schedule: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    """
    Generate a schedule with different permutations for even vs. odd rows of plaquettes.
    
    Args:
        colorcode: ColorCode instance to get qubit coordinates
        initial_schedule: Initial schedule structure
        
    Returns:
        Schedule dictionary with different permutations for even/odd rows
    """
    # Generate two different random orders
    random_order_1 = list(range(6))  # [0, 1, 2, 3, 4, 5]
    random.shuffle(random_order_1)
    
    random_order_2 = list(range(6))  # [0, 1, 2, 3, 4, 5]
    random.shuffle(random_order_2)
    
    print(f"    Even row order: {random_order_1}")
    print(f"    Odd row order: {random_order_2}")
    
    # Create schedules for Z stabilizers
    z_schedules = []
    for stab_idx, anc_qubit in enumerate(colorcode.qubit_groups['anc_Z']):
        y_coord = anc_qubit['y']
        if y_coord % 2 == 0:  # Even row
            z_schedules.append(random_order_1.copy())
        else:  # Odd row
            z_schedules.append(random_order_2.copy())
    
    # Create schedules for X stabilizers
    x_schedules = []
    for stab_idx, anc_qubit in enumerate(colorcode.qubit_groups['anc_X']):
        y_coord = anc_qubit['y']
        if y_coord % 2 == 0:  # Even row
            x_schedules.append(random_order_1.copy())
        else:  # Odd row
            x_schedules.append(random_order_2.copy())
    
    return {
        'Z': z_schedules,
        'X': x_schedules,
    }


if __name__ == "__main__":
    # Run simple optimization with exhaustive search and logical error evaluation
    best_dist, best_sched, best_error_rate = simple_distance_optimization(
        d=7,                    # Code distance
        n_samples=50,           # Number of random schedules to try (ignored if exhaustive_search=True)
        p_cnot=1e-2,           # CNOT error probability
        use_row_based_permutations=False,  # Enable different permutations for even/odd rows
        exhaustive_search=True,  # Enable exhaustive search of all permutations starting with 0
        evaluate_logical_errors=True,      # Enable logical error rate evaluation
        n_logical_shots=1000    # Number of shots for logical error rate evaluation (reduced for speed)
    )
    
    print(f"\nFinal result: Best distance = {best_dist}, Best logical error rate = {best_error_rate:.2e}")
