#!/usr/bin/env python3

import sys
import os
import random
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from color_code_modified import ColorCode

class CorrelatedErrorOptimizer:
    """Optimizer for correlated error configurations in color codes."""
    
    def __init__(self, d: int, circuit_type: str = "tri", seed: Optional[int] = None):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        d : int
            Code distance
        circuit_type : str
            Type of color code circuit
        seed : int, optional
            Random seed for reproducibility
        """
        self.d = d
        self.circuit_type = circuit_type
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Offset directions mapping
        self.offsets = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
        
        # Create initial color code to get ancilla qubit information
        self._create_initial_code()
        
    def _create_initial_code(self):
        """Create initial color code to extract ancilla qubit and data qubit info."""
        temp_code = ColorCode(
            d=self.d,
            circuit_type=self.circuit_type,
            cnot_schedule="tri_optimal",
            correlated_errors={},
            p_correlated=0.0
        )
        # Get only Z ancilla qubits (X ancillas share the same data qubits)
        self.ancilla_qubits = {}
        for vertex in temp_code.tanner_graph.vs:
            if vertex["pauli"] == "Z":  # Only Z ancillas
                name = f"{vertex['x']}-{vertex['y']}-{vertex['pauli']}"
                self.ancilla_qubits[name] = {
                    'x': vertex['x'],
                    'y': vertex['y'],
                    'pauli': vertex['pauli'],
                    'color': vertex['color']
                }
        # Build (x, y) -> data qubit index mapping
        self.data_pos_to_index = {}
        for vertex in temp_code.tanner_graph.vs:
            if vertex["pauli"] is None:
                self.data_pos_to_index[(vertex["x"], vertex["y"])] = vertex.index
        print(f"Found {len(self.ancilla_qubits)} ancilla qubits")
        print(f"Found {len(self.data_pos_to_index)} data qubits")
        
    def get_valid_offsets_for_ancilla(self, ancilla_name: str) -> List[int]:
        """
        Get the valid offset directions for a given ancilla qubit.
        
        Parameters:
        -----------
        ancilla_name : str
            Name of the ancilla qubit
            
        Returns:
        --------
        list
            List of valid offset indices (0-5) for this ancilla
        """
        ancilla_info = self.ancilla_qubits[ancilla_name]
        ancilla_x, ancilla_y = ancilla_info['x'], ancilla_info['y']
        
        valid_offsets = []
        for offset_idx in range(6):
            dx, dy = self.offsets[offset_idx]
            data_x = ancilla_x + dx
            data_y = ancilla_y + dy
            
            # Check if this position corresponds to an existing data qubit
            if (data_x, data_y) in self.data_pos_to_index:
                valid_offsets.append(offset_idx)
        
        return valid_offsets
    
    def generate_random_configuration(self, pairs_per_plaquette: int = 1) -> Dict[str, List[Set[int]]]:
        """
        Generate a random configuration of correlated errors for all plaquettes.
        
        Parameters:
        -----------
        pairs_per_plaquette : int, default 1
            Number of non-overlapping correlated error pairs per plaquette
            
        Returns:
        --------
        dict
            Random correlated error configuration for all ancilla qubits
        """
        ancilla_names = list(self.ancilla_qubits.keys())
        
        correlated_errors = {}
        skipped_count = 0
        for ancilla_name in ancilla_names:
            # Get valid offset directions for this ancilla
            valid_offsets = self.get_valid_offsets_for_ancilla(ancilla_name)
            
            if len(valid_offsets) >= 2 * pairs_per_plaquette:
                # Generate multiple non-overlapping pairs
                offset_pairs = []
                available_offsets = valid_offsets.copy()
                
                for _ in range(pairs_per_plaquette):
                    if len(available_offsets) >= 2:
                        # Pick a random pair from available offsets
                        offset_pair = set(random.sample(available_offsets, 2))
                        offset_pairs.append(offset_pair)
                        
                        # Remove the used offsets to ensure no overlap
                        for offset in offset_pair:
                            available_offsets.remove(offset)
                    else:
                        break
                
                if len(offset_pairs) == pairs_per_plaquette:
                    correlated_errors[ancilla_name] = offset_pairs
                else:
                    print(f"Warning: Ancilla {ancilla_name} couldn't generate {pairs_per_plaquette} pairs, got {len(offset_pairs)}")
                    skipped_count += 1
            else:
                # Skip this ancilla if it doesn't have enough valid offsets
                print(f"Warning: Ancilla {ancilla_name} has only {len(valid_offsets)} valid offsets, need {2 * pairs_per_plaquette}, skipping")
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} ancilla qubits due to insufficient valid offsets")
            
        return correlated_errors
    
    def get_data_qubit_indices(self, ancilla_name: str, offset_indices: set) -> set:
        """Get the data qubit indices for given ancilla and offset indices."""
        ancilla_info = self.ancilla_qubits[ancilla_name]
        ancilla_x, ancilla_y = ancilla_info['x'], ancilla_info['y']
        indices = set()
        for offset_idx in offset_indices:
            if 0 <= offset_idx < 6:
                dx, dy = self.offsets[offset_idx]
                data_x = ancilla_x + dx
                data_y = ancilla_y + dy
                idx = self.data_pos_to_index.get((data_x, data_y))
                if idx is not None:
                    indices.add(idx)
        return indices
    
    def find_correlated_errors_in_detector_error(self, error_explanation, correlated_errors: Dict[str, List[Set[int]]]) -> List[str]:
        """
        Find which plaquettes' correlated errors are involved in a given detector error.
        """
        involved_plaquettes = []
        # Get the qubits involved in this error using the same logic as get_error_qubits
        error_qubits = self.get_error_qubits(error_explanation)
        
        # For each correlated error, check if its data qubit indices overlap with the error
        for ancilla_name, offset_sets in correlated_errors.items():
            for offset_set in offset_sets:
                error_indices = self.get_data_qubit_indices(ancilla_name, offset_set)
                # If at least 2 qubits overlap, consider this plaquette involved
                if len(error_indices & error_qubits) == len(error_qubits):
                    involved_plaquettes.append(ancilla_name)
        return list(set(involved_plaquettes))
    
    def modify_correlated_error(self, ancilla_name: str, current_offset_sets: List[Set[int]], pair_index: Optional[int] = None) -> List[Set[int]]:
        """
        Modify a correlated error pair to act on different qubits.
        
        Parameters:
        -----------
        ancilla_name : str
            Name of the ancilla qubit
        current_offset_sets : list
            Current list of offset sets for this ancilla
        pair_index : int, optional
            Index of the specific pair to modify. If None, chooses randomly.
            
        Returns:
        --------
        list
            New list of offset sets
        """
        # If no specific pair is specified, choose one randomly
        if pair_index is None:
            pair_index = random.randrange(len(current_offset_sets))
        else:
            pair_index = int(pair_index)  # Ensure it's an int
        
        # Get all currently used offsets
        used_offsets = set()
        for i, offset_set in enumerate(current_offset_sets):
            if i != pair_index:  # Don't include the pair we're modifying
                used_offsets.update(offset_set)
        
        # Get valid offset directions for this ancilla
        all_valid_offsets = self.get_valid_offsets_for_ancilla(ancilla_name)
        
        # Remove used offsets to ensure no overlap
        available_offsets = [offset for offset in all_valid_offsets if offset not in used_offsets]
        
        if len(available_offsets) >= 2:
            # Pick a new random pair from available valid offsets
            new_offset_pair = set(random.sample(available_offsets, 2))
        else:
            # Fallback: keep current offsets if no alternatives
            print(f"Warning: Ancilla {ancilla_name} has insufficient available offsets for pair {pair_index}, keeping current")
            new_offset_pair = current_offset_sets[pair_index]
        
        # Create new list with the modified pair
        new_offset_sets = current_offset_sets.copy()
        new_offset_sets[pair_index] = new_offset_pair
        
        return new_offset_sets
    
    def get_error_qubits(self, error_explanation) -> set:
        """
        Get the set of qubits for the circuit error location with the smallest number of qubits.
        If there's a tie, choose randomly from among the locations with the same number of qubits.
        
        Parameters:
        -----------
        error_explanation
            Error explanation from search_for_undetectable_logical_errors
            
        Returns:
        --------
        set
            Set of qubit indices from the chosen circuit error location
        """
        if not error_explanation.circuit_error_locations:
            return set()
        
        # Get qubit sets for each circuit error location
        qubit_sets = []
        for circuit_error in error_explanation.circuit_error_locations:
            targets = circuit_error.flipped_pauli_product
            qubit_indices = set(target.gate_target.value for target in targets)
            qubit_sets.append(qubit_indices)
        
        # Find the minimum size
        min_size = min(len(qubit_set) for qubit_set in qubit_sets)
        
        # Find all locations with the minimum size
        min_size_locations = [i for i, qubit_set in enumerate(qubit_sets) if len(qubit_set) == min_size]
        
        # Choose randomly from the minimum size locations
        chosen_location = random.choice(min_size_locations)
        
        return qubit_sets[chosen_location]
    
    def optimize_correlated_errors(self, 
                                 initial_config: Dict[str, List[Set[int]]],
                                 max_iterations: int = 10,
                                 p_correlated: float = 0.1,
                                 p_bitflip: float = 0.05) -> Tuple[Dict[str, List[Set[int]]], List[int]]:
        """
        Optimize correlated error configuration.
        
        Parameters:
        -----------
        initial_config : dict
            Initial correlated error configuration
        max_iterations : int
            Maximum number of optimization iterations
        p_correlated : float
            Probability of correlated errors
        p_bitflip : float
            Probability of bit-flip errors
            
        Returns:
        --------
        tuple
            (optimized_config, iteration_history)
        """
        current_config = initial_config.copy()
        iteration_history = []
        
        print(f"Starting optimization with {len(current_config)} correlated errors")
        print(f"Max iterations: {max_iterations}")
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Create color code with current configuration
            colorcode = ColorCode(
                d=self.d,
                circuit_type=self.circuit_type,
                cnot_schedule="tri_optimal",
                correlated_errors=current_config,
                p_correlated=p_correlated,
                p_bitflip=p_bitflip
            )
            
            # Find undetectable logical errors
            print("Searching for undetectable logical errors...")
            undetectable_errors = colorcode.circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=9999,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False,
                canonicalize_circuit_errors=False
            )
            
            print(f"Circuit level distance: {len(undetectable_errors)}")
            
            # Collect qubits involved in single-qubit and multi-qubit errors
            single_qubit_error_qubits = set()
            multi_qubit_error_qubits = set()
            
            for error in undetectable_errors:
                error_qubits = self.get_error_qubits(error)
                if len(error_qubits) == 1:
                    single_qubit_error_qubits.update(error_qubits)
                elif len(error_qubits) > 1:
                    multi_qubit_error_qubits.update(error_qubits)
            
            # Remove qubits that appear in both sets (prioritize multi-qubit)
            single_qubit_error_qubits -= multi_qubit_error_qubits
            
            # Convert qubit indices to positions for highlighting
            single_qubit_positions = []
            multi_qubit_positions = []
            
            for qubit_idx in single_qubit_error_qubits:
                # Find the vertex with this index
                for vertex in colorcode.tanner_graph.vs:
                    if vertex.index == qubit_idx and vertex["pauli"] is None:  # Data qubits only
                        single_qubit_positions.append((vertex["x"], vertex["y"]))
                        break
            
            for qubit_idx in multi_qubit_error_qubits:
                # Find the vertex with this index
                for vertex in colorcode.tanner_graph.vs:
                    if vertex.index == qubit_idx and vertex["pauli"] is None:  # Data qubits only
                        multi_qubit_positions.append((vertex["x"], vertex["y"]))
                        break
            
            # Draw the lattice
            fig, ax = plt.subplots(figsize=(10, 8))
            colorcode.draw_lattice(
                ax=ax,
                highlight_qubits=single_qubit_positions,  # Single-qubit errors
                highlight_qubits2=multi_qubit_positions,  # Multi-qubit errors
                highlight_qubit_color='orange',
                highlight_qubit_color2='red',
                highlight_qubit_marker='o',
                highlight_qubit_marker2='s'
            )
            
            # Add black lines for correlated errors
            for ancilla_name, offset_sets in current_config.items():
                for offset_set in offset_sets:
                    # Get the two qubit positions for this correlated error
                    qubit_positions = []
                    ancilla_info = self.ancilla_qubits[ancilla_name]
                    ancilla_x, ancilla_y = ancilla_info['x'], ancilla_info['y']
                    
                    for offset_idx in offset_set:
                        if 0 <= offset_idx < 6:
                            dx, dy = self.offsets[offset_idx]
                            data_x = ancilla_x + dx
                            data_y = ancilla_y + dy
                            qubit_positions.append((data_x, data_y))
                    
                    # Draw line between the two qubits if we have both positions
                    if len(qubit_positions) == 2:
                        x_coords = [qubit_positions[0][0], qubit_positions[1][0]]
                        y_coords = [qubit_positions[0][1], qubit_positions[1][1]]
                        ax.plot(x_coords, y_coords, 'k-', linewidth=2, alpha=0.7)
            
            ax.set_title(f'Iteration {iteration + 1} - Correlated Errors and Error Patterns, Distance: {len(undetectable_errors)}')
            plt.tight_layout()
            plt.show()
            
            if not undetectable_errors:
                print("No undetectable errors found! Optimization complete.")
                break
            
            # Find errors where ALL circuit error locations act on more than one qubit
            multi_qubit_errors = []
            print("Analyzing undetectable errors:")
            for i, error in enumerate(undetectable_errors):
                error_qubits = self.get_error_qubits(error)
                print(f"  Error {i+1}: qubits = {error_qubits}")
                if len(error_qubits) > 1:
                    multi_qubit_errors.append(error)
                    print(f"    -> Multi-qubit error, added to multi-qubit errors")
            
            print(f"Found {len(multi_qubit_errors)} multi-qubit errors")
            
            if not multi_qubit_errors:
                print("No multi-qubit errors found. Optimization complete.")
                break
            
            # Find which plaquettes these errors belong to
            involved_plaquettes = []
            for error in multi_qubit_errors:
                plaquettes = self.find_correlated_errors_in_detector_error(error, current_config)
                involved_plaquettes.extend(plaquettes)
            
            involved_plaquettes = list(set(involved_plaquettes))  # Remove duplicates
            print(f"Errors involve {len(involved_plaquettes)} plaquettes: {involved_plaquettes}")
            
            if not involved_plaquettes:
                print("No plaquettes identified. Optimization complete.")
                break
            
            # Choose one plaquette at random
            chosen_plaquette = random.choice(involved_plaquettes)
            
            print(f"Chosen plaquette for modification: {chosen_plaquette}")
            
            # Modify the correlated error in the chosen plaquette
            current_offset_sets = current_config[chosen_plaquette]  # List of offset sets
            new_offset_sets = self.modify_correlated_error(chosen_plaquette, current_offset_sets)
            
            print(f"Modifying {chosen_plaquette}: {current_offset_sets} -> {new_offset_sets}")
            
            # Update configuration
            current_config[chosen_plaquette] = new_offset_sets

            # # overwrite by totally randomizing all plaquettes by calling generate_random_configuration
            # current_config = self.generate_random_configuration(pairs_per_plaquette=1)
            
            # Record the number of undetectable errors for this iteration
            iteration_history.append(len(undetectable_errors))
            
            print(f"Iteration {iteration + 1} complete. Undetectable errors: {len(undetectable_errors)}")
        
        print(f"\nOptimization complete after {len(iteration_history)} iterations")
        return current_config, iteration_history

def main():
    """Main function to run the optimization."""
    
    # Parameters
    d = 11
    circuit_type = "tri"
    max_iterations = 50
    p_correlated = 0.1
    p_bitflip = 0.05
    seed = 42
    
    print("=== Correlated Error Optimization ===")
    print(f"Code distance: {d}")
    print(f"Circuit type: {circuit_type}")
    print(f"Max iterations: {max_iterations}")
    print(f"Seed: {seed}")
    print()
    
    # Initialize optimizer
    optimizer = CorrelatedErrorOptimizer(d, circuit_type, seed)
    
    # Generate initial random configuration
    pairs_per_plaquette = 1  # Can be increased for more complex patterns
    print(f"Generating initial random configuration with {pairs_per_plaquette} pairs per plaquette...")
    initial_config = optimizer.generate_random_configuration(pairs_per_plaquette)
    
    print("Initial configuration:")
    for ancilla_name, offset_sets in initial_config.items():
        print(f"  {ancilla_name}: {list(offset_sets[0])}")
    print()
    
    # Run optimization
    optimized_config, history = optimizer.optimize_correlated_errors(
        initial_config, max_iterations, p_correlated, p_bitflip
    )
    
    # Results
    print("\n=== Optimization Results ===")
    print(f"Final configuration:")
    for ancilla_name, offset_sets in optimized_config.items():
        print(f"  {ancilla_name}: {list(offset_sets[0])}")
    
    print(f"\nIteration history (number of undetectable errors):")
    print(history)
    
    if history:
        print(f"Initial undetectable errors: {history[0]}")
        print(f"Final undetectable errors: {history[-1]}")
        print(f"Improvement: {history[0] - history[-1]} errors")
        # plot the history
        plt.plot(history)
        plt.show()
    
    return optimized_config, history

if __name__ == "__main__":
    main() 