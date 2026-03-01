import random
import numpy as np
import pickle
import time
from typing import Dict, List, Tuple, Optional
from color_code_modified import ColorCode
from color_code_cnot_optimization import get_initial_cnot_schedule_dict, build_color_code_with_schedule


class DistanceBasedCNOTOptimizer:
    """
    Optimizes CNOT schedules to maximize code distance by sampling random schedules
    and using a two-tier distance calculation approach.
    """
    
    def __init__(self, d: int, p_cnot: float = 1e-3, 
                 tri_optimal_schedule: Optional[List[int]] = None,
                 init_option: str = 'benchmark'):
        """
        Initialize the optimizer.
        
        Args:
            d: Code distance
            p_cnot: CNOT error probability
            tri_optimal_schedule: Base tri-optimal schedule
            init_option: Initialization option for CNOT schedules
        """
        self.d = d
        self.rounds = 2  # Hardcoded to 2 as per ColorCode implementation
        self.p_cnot = p_cnot
        self.tri_optimal_schedule = tri_optimal_schedule or [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
        self.init_option = init_option
        
        # Build initial ColorCode to get stabilizer counts
        self.initial_colorcode = ColorCode(
            d=d, cnot_schedule="tri_optimal", p_bitflip=p_cnot
        )
        
        # Get initial CNOT schedule dictionary
        self.initial_schedule = get_initial_cnot_schedule_dict(
            self.initial_colorcode, self.tri_optimal_schedule, init_option
        )
        
        # Optimization history
        self.history = []
        self.best_distance = 0
        self.best_schedule = None
        
    def get_random_schedule(self, schedule_type: str = 'uniform') -> Dict[str, List[List[int]]]:
        """
        Generate a random CNOT schedule.
        
        Args:
            schedule_type: Type of randomization
                - 'uniform': Same random order for all stabilizers of each type
                - 'individual': Different random order for each stabilizer
                - 'mixed': Mix of uniform and individual randomization
        
        Returns:
            Dictionary with 'X' and 'Z' stabilizer schedules
        """
        if schedule_type == 'uniform':
            # Same random order for all X stabilizers, same for all Z stabilizers
            x_order = list(range(6))
            z_order = list(range(6))
            random.shuffle(x_order)
            random.shuffle(z_order)
            
            x_schedules = [x_order.copy() for _ in self.initial_schedule['X']]
            z_schedules = [z_order.copy() for _ in self.initial_schedule['Z']]
            
        elif schedule_type == 'individual':
            # Different random order for each stabilizer
            x_schedules = []
            z_schedules = []
            
            for _ in self.initial_schedule['X']:
                order = list(range(6))
                random.shuffle(order)
                x_schedules.append(order)
                
            for _ in self.initial_schedule['Z']:
                order = list(range(6))
                random.shuffle(order)
                z_schedules.append(order)
                
        elif schedule_type == 'mixed':
            # Randomly choose between uniform and individual for each type
            x_schedules = self.get_random_schedule('uniform' if random.random() < 0.5 else 'individual')['X']
            z_schedules = self.get_random_schedule('uniform' if random.random() < 0.5 else 'individual')['Z']
            
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
            
        return {'X': x_schedules, 'Z': z_schedules}
    
    def compute_graphlike_distance(self, circuit) -> int:
        """
        Compute the graphlike distance as a fast upper bound.
        
        Args:
            circuit: Stim circuit
            
        Returns:
            Graphlike distance
        """
        try:
            return len(circuit.shortest_graphlike_error())
        except Exception as e:
            print(f"Warning: Could not compute graphlike distance: {e}")
            return 0
    
    def compute_accurate_distance(self, circuit) -> int:
        """
        Compute the accurate distance using search_for_undetectable_logical_errors.
        
        Args:
            circuit: Stim circuit
            
        Returns:
            Accurate distance
        """
        try:
            undetectable_errors = circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=4,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False,
                canonicalize_circuit_errors=False
            )
            return len(undetectable_errors)
        except Exception as e:
            print(f"Warning: Could not compute accurate distance: {e}")
            return 0
    
    def evaluate_schedule(self, schedule: Dict[str, List[List[int]]], 
                         use_fast_check: bool = True) -> Tuple[int, float]:
        """
        Evaluate a CNOT schedule by computing its distance.
        
        Args:
            schedule: CNOT schedule dictionary
            use_fast_check: Whether to use fast graphlike distance check first
            
        Returns:
            Tuple of (distance, computation_time)
        """
        start_time = time.time()
        
        try:
            # Build ColorCode with the schedule
            colorcode = build_color_code_with_schedule(
                self.d, schedule, self.p_cnot
            )
            circuit = colorcode.circuit
            
            if use_fast_check:
                # First compute graphlike distance as upper bound
                graphlike_distance = self.compute_graphlike_distance(circuit)
                
                # If graphlike distance is lower than best, skip accurate computation
                if graphlike_distance <= self.best_distance:
                    computation_time = time.time() - start_time
                    return graphlike_distance, computation_time
            
            # Compute accurate distance
            accurate_distance = self.compute_accurate_distance(circuit)
            computation_time = time.time() - start_time
            
            return accurate_distance, computation_time
            
        except Exception as e:
            print(f"Error evaluating schedule: {e}")
            computation_time = time.time() - start_time
            return 0, computation_time
    
    def optimize(self, n_samples: int = 100, schedule_types: List[str] = None,
                save_history: bool = True, history_file: str = 'distance_optimization_history.pkl') -> Dict:
        """
        Run the distance-based optimization.
        
        Args:
            n_samples: Number of random schedules to sample
            schedule_types: List of schedule types to try
            save_history: Whether to save optimization history
            history_file: File to save history to
            
        Returns:
            Dictionary with optimization results
        """
        if schedule_types is None:
            schedule_types = ['uniform', 'individual', 'mixed']
        
        print(f"Starting distance-based CNOT optimization...")
        print(f"Code parameters: d={self.d}, rounds={self.rounds}, p_cnot={self.p_cnot}")
        print(f"Will sample {n_samples} schedules using types: {schedule_types}")
        print(f"Initial best distance: {self.best_distance}")
        print("-" * 60)
        
        total_time = time.time()
        fast_checks = 0
        accurate_checks = 0
        
        for sample_idx in range(n_samples):
            # Choose random schedule type
            schedule_type = random.choice(schedule_types)
            
            # Generate random schedule
            schedule = self.get_random_schedule(schedule_type)
            
            print(f"\nSample {sample_idx + 1}/{n_samples} (type: {schedule_type})")
            
            # Evaluate schedule with fast check first
            distance, comp_time = self.evaluate_schedule(schedule, use_fast_check=True)
            
            if distance > 0:  # Valid schedule
                if distance > self.best_distance:
                    print(f"  NEW BEST! Distance: {distance} (previous: {self.best_distance})")
                    self.best_distance = distance
                    self.best_schedule = schedule.copy()
                    
                    # Record in history
                    self.history.append({
                        'sample_idx': sample_idx,
                        'schedule_type': schedule_type,
                        'distance': distance,
                        'computation_time': comp_time,
                        'schedule': schedule.copy(),
                        'is_best': True
                    })
                else:
                    print(f"  Distance: {distance} (best: {self.best_distance})")
                    self.history.append({
                        'sample_idx': sample_idx,
                        'schedule_type': schedule_type,
                        'distance': distance,
                        'computation_time': comp_time,
                        'schedule': schedule.copy(),
                        'is_best': False
                    })
                
                if distance == self.best_distance:
                    accurate_checks += 1
                else:
                    fast_checks += 1
            else:
                print(f"  Invalid schedule (distance: {distance})")
                self.history.append({
                    'sample_idx': sample_idx,
                    'schedule_type': schedule_type,
                    'distance': distance,
                    'computation_time': comp_time,
                    'schedule': schedule.copy(),
                    'is_best': False
                })
        
        total_time = time.time() - total_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total samples: {n_samples}")
        print(f"Best distance found: {self.best_distance}")
        print(f"Fast checks (skipped): {fast_checks}")
        print(f"Accurate checks: {accurate_checks}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per sample: {total_time/n_samples:.3f}s")
        
        # Save history if requested
        if save_history:
            self.save_history(history_file)
            print(f"History saved to: {history_file}")
        
        return {
            'best_distance': self.best_distance,
            'best_schedule': self.best_schedule,
            'total_samples': n_samples,
            'fast_checks': fast_checks,
            'accurate_checks': accurate_checks,
            'total_time': total_time,
            'history': self.history
        }
    
    def save_history(self, filename: str):
        """Save optimization history to file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'best_distance': self.best_distance,
                'best_schedule': self.best_schedule,
                'history': self.history,
                'parameters': {
                    'd': self.d,
                    'rounds': self.rounds,
                    'p_cnot': self.p_cnot,
                    'init_option': self.init_option
                }
            }, f)
    
    def load_history(self, filename: str):
        """Load optimization history from file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.best_distance = data['best_distance']
                self.best_schedule = data['best_schedule']
                self.history = data['history']
                print(f"Loaded history from {filename}")
                print(f"Best distance: {self.best_distance}")
                print(f"History entries: {len(self.history)}")
        except FileNotFoundError:
            print(f"History file {filename} not found.")
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def get_best_schedule_summary(self) -> str:
        """Get a summary of the best schedule found."""
        if self.best_schedule is None:
            return "No best schedule found yet."
        
        summary = f"Best distance: {self.best_distance}\n"
        summary += f"X stabilizers: {len(self.best_schedule['X'])}\n"
        summary += f"Z stabilizers: {len(self.best_schedule['Z'])}\n\n"
        
        summary += "X stabilizer schedules:\n"
        for i, schedule in enumerate(self.best_schedule['X']):
            summary += f"  {i}: {schedule}\n"
        
        summary += "\nZ stabilizer schedules:\n"
        for i, schedule in enumerate(self.best_schedule['Z']):
            summary += f"  {i}: {schedule}\n"
        
        return summary


def main():
    """Example usage of the DistanceBasedCNOTOptimizer."""
    
    # Initialize optimizer
    optimizer = DistanceBasedCNOTOptimizer(
        d=7, 
        p_cnot=1e-3,
        init_option='benchmark'
    )
    
    # Run optimization
    results = optimizer.optimize(
        n_samples=50,  # Start with fewer samples for testing
        schedule_types=['uniform', 'individual', 'mixed']
    )
    
    # Print results
    print("\n" + optimizer.get_best_schedule_summary())
    
    # Optionally save the best schedule for later use
    if optimizer.best_schedule:
        with open('best_distance_schedule.pkl', 'wb') as f:
            pickle.dump(optimizer.best_schedule, f)
        print("\nBest schedule saved to 'best_distance_schedule.pkl'")


if __name__ == "__main__":
    main()
