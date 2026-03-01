"""
Targeted Schedule Optimizer for Color Code CNOT Schedules.

Algorithm:
1. Start with random per-plaquette schedules
2. Find problematic error via search_for_undetectable_logical_errors
3. Identify auxiliary qubits in the minimal error representative
4. Choose one auxiliary at random
5. Change its schedule to a new random schedule
6. Accept if: distance increased OR (distance didn't decrease AND error structure changed)
7. Track distance progress over iterations
"""

import random
import itertools
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from color_code_stim import ColorCode


@dataclass
class OptimizationStep:
    """Record of one optimization step."""
    iteration: int
    distance: int
    changed_auxiliary: Optional[int]
    changed_type: Optional[str]  # 'Z' or 'X'
    old_schedule: Optional[List[int]]
    new_schedule: Optional[List[int]]
    time_elapsed: float


class TargetedScheduleOptimizer:
    """
    Optimizes CNOT schedules by targeting qubits involved in undetectable errors.
    """
    
    def __init__(self, d: int, rounds: int = 2, p_cnot: float = 0.0):
        """
        Initialize optimizer.
        
        Args:
            d: Code distance
            rounds: Number of syndrome extraction rounds
            p_cnot: CNOT error probability (0 for distance calculation)
        """
        self.d = d
        self.rounds = rounds
        self.p_cnot = p_cnot
        
        # Build initial ColorCode to get structure
        self.base_colorcode = ColorCode(
            d=d, rounds=rounds, cnot_schedule="tri_optimal", p_cnot=p_cnot
        )
        
        self.num_z_stabs = len(self.base_colorcode.qubit_groups['anc_Z'])
        self.num_x_stabs = len(self.base_colorcode.qubit_groups['anc_X'])
        
        # Map from qid to (type, index)
        self.qid_to_stab = {}
        for i, anc in enumerate(self.base_colorcode.qubit_groups['anc_Z']):
            self.qid_to_stab[anc['qid']] = ('Z', i)
        for i, anc in enumerate(self.base_colorcode.qubit_groups['anc_X']):
            self.qid_to_stab[anc['qid']] = ('X', i)
        
        # All possible schedules (permutations of [0,1,2,3,4,5])
        self.all_schedules = list(itertools.permutations(range(6)))
        
        # Optimization history
        self.history: List[OptimizationStep] = []
        
    def get_random_schedule_dict(self) -> Dict[str, List[List[int]]]:
        """Generate random per-plaquette schedules."""
        z_schedules = [list(random.choice(self.all_schedules)) for _ in range(self.num_z_stabs)]
        x_schedules = [list(random.choice(self.all_schedules)) for _ in range(self.num_x_stabs)]
        return {'Z': z_schedules, 'X': x_schedules}
    
    def build_circuit(self, schedule_dict: Dict[str, List[List[int]]]):
        """Build ColorCode circuit with given schedule."""
        colorcode = ColorCode(
            d=self.d,
            rounds=self.rounds,
            cnot_schedule=schedule_dict,
            p_cnot=self.p_cnot,
        )
        return colorcode.circuit
    
    def find_undetectable_error(self, circuit) -> Tuple[int, List]:
        """
        Find an undetectable logical error.
        
        Returns:
            Tuple of (distance, error_instructions)
        """
        errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=False
        )
        return len(errors), errors
    
    def extract_qubits_from_location(self, loc) -> Set[int]:
        """Extract all qubit IDs from a single CircuitErrorLocation."""
        qids = set()
        if hasattr(loc, 'flipped_pauli_product'):
            for pt in loc.flipped_pauli_product:
                if hasattr(pt, 'gate_target'):
                    gt = pt.gate_target
                    if hasattr(gt, 'value'):
                        qids.add(gt.value)
        return qids
    
    def extract_auxiliary_qubits_from_error(self, errors: List) -> Set[int]:
        """
        Extract auxiliary qubit IDs from the minimal representative of each error.
        
        For each error, finds the circuit_error_location with the fewest qubits
        (the minimal representative), then checks if any of those are auxiliaries.
        
        Args:
            errors: List of ExplainedError objects from search_for_undetectable_logical_errors
            
        Returns:
            Set of auxiliary qubit IDs involved in the minimal error representatives
        """
        aux_qids = set()
        
        for error in errors:
            if not hasattr(error, 'circuit_error_locations'):
                continue
            
            # Find the location with the fewest qubits (minimal representative)
            min_qubits = None
            min_count = float('inf')
            
            for loc in error.circuit_error_locations:
                qids = self.extract_qubits_from_location(loc)
                if len(qids) < min_count:
                    min_count = len(qids)
                    min_qubits = qids
            
            # Check if any qubits in the minimal representative are auxiliaries
            if min_qubits:
                for qid in min_qubits:
                    if qid in self.qid_to_stab:
                        aux_qids.add(qid)
        
        return aux_qids
    
    def optimize(self, max_iterations: int = 100, 
                 max_no_improvement: int = 20,
                 verbose: bool = True) -> Dict:
        """
        Run the targeted optimization using random single-step changes.
        
        Algorithm:
        1. Find all auxiliaries involved in the minimal error
        2. Choose one at random
        3. Change its schedule to a new random schedule
        4. Accept if:
           - Distance increased, OR
           - Distance didn't decrease AND error structure changed (different auxiliaries)
        
        Args:
            max_iterations: Maximum number of iterations
            max_no_improvement: Stop if no improvement for this many iterations
            verbose: Print progress
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Initialize with random schedules
        schedule_dict = self.get_random_schedule_dict()
        circuit = self.build_circuit(schedule_dict)
        
        current_distance, current_errors = self.find_undetectable_error(circuit)
        best_distance = current_distance
        best_schedule = {k: [s.copy() for s in v] for k, v in schedule_dict.items()}
        
        if verbose:
            print(f"Initial distance: {current_distance}")
            print(f"Total Z stabilizers: {self.num_z_stabs}, X stabilizers: {self.num_x_stabs}")
            print(f"Possible schedules per plaquette: {len(self.all_schedules)}")
            print("-" * 60)
        
        # Record initial state
        self.history.append(OptimizationStep(
            iteration=0,
            distance=current_distance,
            changed_auxiliary=None,
            changed_type=None,
            old_schedule=None,
            new_schedule=None,
            time_elapsed=0.0
        ))
        
        iterations_without_improvement = 0
        iteration = 0
        
        while iteration < max_iterations and iterations_without_improvement < max_no_improvement:
            iteration += 1
            iter_start = time.time()
            
            # Find auxiliary qubits in the current error
            aux_qids = self.extract_auxiliary_qubits_from_error(current_errors)
            
            if verbose:
                print(f"\nIteration {iteration}: distance={current_distance}, "
                      f"auxiliaries in error: {len(aux_qids)}")
            
            if not aux_qids:
                if verbose:
                    print("  No auxiliary qubits found in error")
                iterations_without_improvement += 1
                continue
            
            # Filter to only auxiliaries we know about
            valid_aux_qids = [qid for qid in aux_qids if qid in self.qid_to_stab]
            if not valid_aux_qids:
                if verbose:
                    print("  No valid auxiliary qubits found")
                iterations_without_improvement += 1
                continue
            
            # Choose one auxiliary at random
            aux_qid = random.choice(valid_aux_qids)
            stab_type, stab_idx = self.qid_to_stab[aux_qid]
            old_schedule = schedule_dict[stab_type][stab_idx].copy()
            
            # Choose a new random schedule (different from current)
            new_schedule = list(random.choice(self.all_schedules))
            while new_schedule == old_schedule:
                new_schedule = list(random.choice(self.all_schedules))
            
            if verbose:
                print(f"  Trying {stab_type}[{stab_idx}] (qid={aux_qid}): "
                      f"{old_schedule} -> {new_schedule}")
            
            # Apply new schedule
            schedule_dict[stab_type][stab_idx] = new_schedule
            
            # Build circuit and check error
            try:
                circuit = self.build_circuit(schedule_dict)
                new_distance, new_errors = self.find_undetectable_error(circuit)
            except Exception as e:
                # Revert and continue
                schedule_dict[stab_type][stab_idx] = old_schedule
                iterations_without_improvement += 1
                if verbose:
                    print(f"  Error building circuit: {e}")
                continue
            
            # Check if this auxiliary is still in the error
            new_aux_qids = self.extract_auxiliary_qubits_from_error(new_errors)
            
            # Acceptance criteria:
            # 1. Distance increased
            # 2. Distance didn't decrease AND error structure changed (different auxiliaries)
            distance_increased = new_distance > current_distance
            distance_not_decreased = new_distance >= current_distance
            error_structure_changed = new_aux_qids != aux_qids
            
            accept = distance_increased or (distance_not_decreased and error_structure_changed)
            
            if accept:
                # Determine reason for acceptance
                if distance_increased:
                    reason = "distance increased"
                elif aux_qid not in new_aux_qids:
                    reason = "auxiliary removed"
                else:
                    reason = "error structure changed"
                
                if verbose:
                    print(f"  ACCEPTED ({reason})! "
                          f"Distance: {current_distance} -> {new_distance}, "
                          f"Auxiliaries: {len(aux_qids)} -> {len(new_aux_qids)}")
                
                current_distance = new_distance
                current_errors = new_errors
                
                # Track best distance
                if new_distance > best_distance:
                    best_distance = new_distance
                    best_schedule = {k: [s.copy() for s in v] for k, v in schedule_dict.items()}
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
                
                # Record step
                self.history.append(OptimizationStep(
                    iteration=iteration,
                    distance=new_distance,
                    changed_auxiliary=aux_qid,
                    changed_type=stab_type,
                    old_schedule=old_schedule,
                    new_schedule=new_schedule,
                    time_elapsed=time.time() - iter_start
                ))
            else:
                # Reject - revert schedule
                schedule_dict[stab_type][stab_idx] = old_schedule
                iterations_without_improvement += 1
                if verbose:
                    print(f"  REJECTED. Distance: {current_distance} -> {new_distance}, "
                          f"same auxiliaries. ({iterations_without_improvement}/{max_no_improvement})")
        
        total_time = time.time() - start_time
        
        # Print summary
        if verbose:
            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"Final distance: {current_distance}")
            print(f"Best distance: {best_distance}")
            print(f"Total iterations: {iteration}")
            print(f"Total time: {total_time:.2f}s")
        
        return {
            'final_distance': current_distance,
            'best_distance': best_distance,
            'final_schedule': schedule_dict,
            'best_schedule': best_schedule,
            'iterations': iteration,
            'total_time': total_time,
            'history': self.history
        }
    
    def plot_progress(self, save_path: str = None):
        """Plot distance progress over iterations."""
        if not self.history:
            print("No history to plot")
            return
        
        iterations = [h.iteration for h in self.history]
        distances = [h.distance for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, distances, 'b-o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Circuit Distance')
        plt.title(f'Targeted Schedule Optimization (d={self.d})')
        plt.grid(True, alpha=0.3)
        
        # Mark improvements
        improvements = [(h.iteration, h.distance) for h in self.history 
                       if h.changed_auxiliary is not None]
        if improvements:
            imp_iters, imp_dists = zip(*improvements)
            plt.scatter(imp_iters, imp_dists, c='green', s=100, zorder=5, 
                       label='Schedule change')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()


def main():
    """Run the targeted optimizer."""
    
    # Configuration
    d = 11
    rounds = 2
    max_iterations = 1000  # More iterations for stochastic approach
    max_no_improvement = 200  # Allow more attempts without improvement
    p_cnot = 1e-3  # Need non-zero for error search
    
    print(f"Targeted Schedule Optimization for d={d} color code")
    print("=" * 60)
    
    optimizer = TargetedScheduleOptimizer(d=d, rounds=rounds, p_cnot=p_cnot)
    
    results = optimizer.optimize(
        max_iterations=max_iterations,
        max_no_improvement=max_no_improvement,
        verbose=True
    )
    
    # Plot progress
    optimizer.plot_progress(save_path='results/targeted_optimization_progress.png')
    
    # Print best schedule (highest distance achieved)
    print("\nBest Schedule (distance={})".format(results['best_distance']))
    for stab_type in ['Z', 'X']:
        print(f"  {stab_type} stabilizers:")
        for i, sched in enumerate(results['best_schedule'][stab_type]):
            print(f"    {i}: {sched}")


if __name__ == "__main__":
    main()

