"""
Visualize the minimal undetectable logical error found by search_for_undetectable_logical_errors.
"""

import numpy as np
import matplotlib.pyplot as plt
from color_code_stim import ColorCode
import re


def parse_pauli_product(pauli_str):
    """
    Parse a flipped_pauli_product string like 'Y19[coords 8,2]*X25[coords 10,3]'
    Returns list of (pauli, qubit_idx, x, y) tuples.
    """
    # Pattern to match: P<idx>[coords x,y] where P is X, Y, or Z
    pattern = r'([XYZ])(\d+)\[coords\s+([\d.-]+),([\d.-]+)\]'
    matches = re.findall(pattern, pauli_str)
    
    results = []
    for match in matches:
        pauli = match[0]
        qubit_idx = int(match[1])
        x = float(match[2])
        y = float(match[3])
        results.append((pauli, qubit_idx, x, y))
    
    return results


def parse_detectors(error):
    """
    Parse detector information from an ExplainedError.
    
    Returns list of dicts with detector info: {det_id, x, y, round, pauli_type, color}
    Detector coords format: D<id>[coords x,y,round,pauli_type,color]
    """
    error_str = str(error)
    
    # Find dem_error_terms line
    match = re.search(r'dem_error_terms:\s*([^\n]+)', error_str)
    if not match:
        return []
    
    dem_terms = match.group(1)
    
    # Pattern to match: D<id>[coords x,y,round,pauli,color] or L<id> (logical observable)
    # Detector pattern with 5 coordinates
    det_pattern = r'D(\d+)\[coords\s+([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+)\]'
    det_matches = re.findall(det_pattern, dem_terms)
    
    detectors = []
    for match in det_matches:
        detectors.append({
            'det_id': int(match[0]),
            'x': float(match[1]),
            'y': float(match[2]),
            'round': int(float(match[3])),
            'pauli_type': int(float(match[4])),  # 0=X, 1=Y, 2=Z
            'color': int(float(match[5]))  # 0=r, 1=g, 2=b
        })
    
    # Check for logical observable
    has_logical = 'L0' in dem_terms or 'L1' in dem_terms
    
    return detectors, has_logical


def extract_error_locations(errors, use_minimal_representative=True):
    """
    Extract qubit locations involved in the undetectable error.
    
    Args:
        errors: List of ExplainedError objects
        use_minimal_representative: If True, for each error component choose only
            the representative (CircuitErrorLocation) that acts on the fewest qubits
    
    Returns:
        Dict mapping (x, y) to list of (pauli, qubit_idx, error_idx, location_idx)
    """
    all_locations = {}
    
    for error_idx, error in enumerate(errors):
        error_str = str(error)
        
        # Find all CircuitErrorLocation blocks
        locations = error_str.split('CircuitErrorLocation {')[1:]
        
        if use_minimal_representative and locations:
            # Parse all locations and find the one with fewest qubits
            parsed_locations = []
            for loc_idx, loc in enumerate(locations):
                match = re.search(r'flipped_pauli_product:\s*([^\n]+)', loc)
                if match:
                    pauli_product = match.group(1).strip()
                    paulis = parse_pauli_product(pauli_product)
                    parsed_locations.append({
                        'loc_idx': loc_idx,
                        'pauli_product': pauli_product,
                        'paulis': paulis,
                        'num_qubits': len(paulis)
                    })
            
            # Choose the one with minimum qubits
            if parsed_locations:
                min_loc = min(parsed_locations, key=lambda x: x['num_qubits'])
                
                for pauli, qubit_idx, x, y in min_loc['paulis']:
                    key = (x, y)
                    if key not in all_locations:
                        all_locations[key] = []
                    all_locations[key].append({
                        'pauli': pauli,
                        'qubit_idx': qubit_idx,
                        'error_idx': error_idx,
                        'location_idx': min_loc['loc_idx'],
                        'pauli_product': min_loc['pauli_product']
                    })
        else:
            # Original behavior: include all locations
            for loc_idx, loc in enumerate(locations):
                match = re.search(r'flipped_pauli_product:\s*([^\n]+)', loc)
                if match:
                    pauli_product = match.group(1).strip()
                    paulis = parse_pauli_product(pauli_product)
                    
                    for pauli, qubit_idx, x, y in paulis:
                        key = (x, y)
                        if key not in all_locations:
                            all_locations[key] = []
                        all_locations[key].append({
                            'pauli': pauli,
                            'qubit_idx': qubit_idx,
                            'error_idx': error_idx,
                            'location_idx': loc_idx
                        })
    
    return all_locations


def get_minimal_representative(error):
    """Get the minimal representative (fewest qubits) for a single error, including detectors."""
    error_str = str(error)
    locations = error_str.split('CircuitErrorLocation {')[1:]
    
    parsed_locations = []
    for loc_idx, loc in enumerate(locations):
        match = re.search(r'flipped_pauli_product:\s*([^\n]+)', loc)
        if match:
            pauli_product = match.group(1).strip()
            paulis = parse_pauli_product(pauli_product)
            parsed_locations.append({
                'loc_idx': loc_idx,
                'pauli_product': pauli_product,
                'paulis': paulis,
                'num_qubits': len(paulis)
            })
    
    if parsed_locations:
        min_rep = min(parsed_locations, key=lambda x: x['num_qubits'])
        # Add detector info
        detectors, has_logical = parse_detectors(error)
        min_rep['detectors'] = detectors
        min_rep['has_logical'] = has_logical
        return min_rep
    return None


def visualize_minimal_error(colorcode, errors, title_suffix=""):
    """
    Visualize the minimal undetectable logical error on the color code lattice.
    """
    # Extract error locations
    error_locations = extract_error_locations(errors)
    
    # Draw the base lattice
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = colorcode.draw_lattice(ax=ax, show_data_qubits=True)
    
    # Color scheme for different Paulis
    pauli_colors = {
        'X': 'red',
        'Y': 'purple', 
        'Z': 'blue'
    }
    
    # Plot error locations
    for (x, y), infos in error_locations.items():
        # Determine primary Pauli type (most frequent or first)
        paulis = [info['pauli'] for info in infos]
        pauli_counts = {}
        for p in paulis:
            pauli_counts[p] = pauli_counts.get(p, 0) + 1
        primary_pauli = max(pauli_counts, key=pauli_counts.get)
        
        color = pauli_colors.get(primary_pauli, 'black')
        
        # Draw a larger marker for error locations
        ax.scatter([x], [y], c=color, s=400, marker='o', 
                  edgecolors='black', linewidths=3, zorder=20, alpha=0.8)
        
        # Add Pauli label
        pauli_label = '/'.join(sorted(set(paulis)))
        ax.annotate(pauli_label, (x, y), fontsize=10, fontweight='bold',
                   ha='center', va='center', color='white', zorder=21)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='red', s=200, marker='o', edgecolors='black', linewidths=2, label='X error'),
        plt.scatter([], [], c='purple', s=200, marker='o', edgecolors='black', linewidths=2, label='Y error'),
        plt.scatter([], [], c='blue', s=200, marker='o', edgecolors='black', linewidths=2, label='Z error'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(f'Minimal Undetectable Logical Error (distance={len(errors)}){title_suffix}', fontsize=14)
    
    plt.tight_layout()
    return fig, ax


def visualize_errors_separately(colorcode, errors, title_prefix="", save_prefix="error"):
    """
    Visualize each error component in a separate plot.
    
    Args:
        colorcode: ColorCode instance
        errors: List of ExplainedError objects
        title_prefix: Prefix for plot titles
        save_prefix: Prefix for saved file names
    
    Returns:
        List of (fig, ax) tuples
    """
    pauli_colors = {
        'X': 'red',
        'Y': 'purple', 
        'Z': 'blue'
    }
    
    figures = []
    
    for error_idx, error in enumerate(errors):
        # Get minimal representative for this error
        min_rep = get_minimal_representative(error)
        if min_rep is None:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = colorcode.draw_lattice(ax=ax, show_data_qubits=True)
        
        # Plot error locations for this error only
        for pauli, qubit_idx, x, y in min_rep['paulis']:
            color = pauli_colors.get(pauli, 'black')
            
            # Draw marker
            ax.scatter([x], [y], c=color, s=500, marker='o', 
                      edgecolors='black', linewidths=3, zorder=20, alpha=0.8)
            
            # Add Pauli label
            ax.annotate(pauli, (x, y), fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white', zorder=21)
        
        # Plot triggered detectors with X markers
        detectors = min_rep.get('detectors', [])
        for det in detectors:
            det_x, det_y = det['x'], det['y']
            ax.scatter([det_x], [det_y], c='darkgreen', s=500, marker='x',
                      linewidths=4, zorder=19)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='red', s=200, marker='o', edgecolors='black', linewidths=2, label='X error'),
            plt.scatter([], [], c='purple', s=200, marker='o', edgecolors='black', linewidths=2, label='Y error'),
            plt.scatter([], [], c='blue', s=200, marker='o', edgecolors='black', linewidths=2, label='Z error'),
            plt.scatter([], [], c='darkgreen', s=200, marker='x', linewidths=3, label='Detector triggered'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Title with error and detector details
        det_ids = [f"D{d['det_id']}" for d in detectors]
        has_logical = min_rep.get('has_logical', False)
        det_str = ', '.join(det_ids) if det_ids else 'none'
        if has_logical:
            det_str += ' + L0'
        
        ax.set_title(f'{title_prefix} Error {error_idx + 1}/{len(errors)}\n{min_rep["pauli_product"]}\nDetectors: {det_str}', 
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{error_idx + 1}.png', dpi=150)
        plt.show()
        
        figures.append((fig, ax))
    
    return figures


def visualize_errors_grid(colorcode, errors, title_prefix="", save_name="errors_grid.png"):
    """
    Visualize all error components in a grid of subplots.
    
    Args:
        colorcode: ColorCode instance
        errors: List of ExplainedError objects
        title_prefix: Prefix for plot title
        save_name: Name for saved file
    
    Returns:
        fig, axes
    """
    pauli_colors = {
        'X': 'red',
        'Y': 'purple', 
        'Z': 'blue'
    }
    
    n_errors = len(errors)
    if n_errors == 0:
        return None, None
    
    # Calculate grid dimensions
    n_cols = min(3, n_errors)
    n_rows = (n_errors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Flatten axes for easy iteration
    if n_errors == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    
    for error_idx, error in enumerate(errors):
        ax = axes[error_idx]
        
        # Get minimal representative for this error
        min_rep = get_minimal_representative(error)
        if min_rep is None:
            ax.set_visible(False)
            continue
        
        # Draw lattice
        ax = colorcode.draw_lattice(ax=ax, show_data_qubits=True)
        
        # Plot error locations (data qubits with Pauli errors)
        for pauli, qubit_idx, x, y in min_rep['paulis']:
            color = pauli_colors.get(pauli, 'black')
            
            ax.scatter([x], [y], c=color, s=300, marker='o', 
                      edgecolors='black', linewidths=2, zorder=20, alpha=0.8)
            
            ax.annotate(pauli, (x, y), fontsize=10, fontweight='bold',
                       ha='center', va='center', color='white', zorder=21)
        
        # Plot triggered detectors with X markers
        detectors = min_rep.get('detectors', [])
        for det in detectors:
            det_x, det_y = det['x'], det['y']
            # Use different colors for different detector types
            det_color = 'darkgreen'
            ax.scatter([det_x], [det_y], c=det_color, s=400, marker='x',
                      linewidths=3, zorder=19)
        
        # Build title with detector info
        det_ids = [f"D{d['det_id']}" for d in detectors]
        has_logical = min_rep.get('has_logical', False)
        det_str = ', '.join(det_ids) if det_ids else 'none'
        if has_logical:
            det_str += ' + L0'
        
        ax.set_title(f'Error {error_idx + 1}: {min_rep["pauli_product"]}\nDetectors: {det_str}', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_errors, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{title_prefix} - {n_errors} Error Components', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.show()
    
    return fig, axes


def compare_minimal_errors(d=7, rounds=7, p_cnot=1e-3):
    """
    Compare minimal errors between benchmark and custom schedules.
    """
    from compare_schedules import schedule_to_cnot_dict, schedule_to_cnot_dict_by_color_and_type
    
    # Benchmark schedule
    benchmark_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
    
    # Custom per-color schedules for BULK plaquettes (6 data qubits)
    bulk_schedules = {
        'r': [1, 4, 2, 6, 3, 5],
        'b': [1, 5, 3, 6, 2, 4],
        'g': [4, 2, 6, 3, 5, 1],
    }
    for color in ['r', 'g', 'b']:
        bulk_schedules[color] = bulk_schedules[color] + [x + 6 for x in bulk_schedules[color]]
    
    # Custom per-color schedules for EDGE plaquettes (4 data qubits)
    edge_schedules = {
        'r': [3, 2, 6, 4, 5, 1],
        'b': [6, 4, 5, 1, 3, 2],
        'g': [5, 1, 3, 2, 6, 4],
    }
    for color in ['r', 'g', 'b']:
        edge_schedules[color] = edge_schedules[color] + [x + 6 for x in edge_schedules[color]]
    
    # Combine into bulk/edge structure
    custom_schedules_by_color_and_type = {
        color: {'bulk': bulk_schedules[color], 'edge': edge_schedules[color]}
        for color in ['r', 'g', 'b']
    }

    # Build benchmark code
    colorcode_init = ColorCode(d=d, rounds=rounds, cnot_schedule="tri_optimal", p_cnot=p_cnot)
    benchmark_dict = schedule_to_cnot_dict(colorcode_init, benchmark_schedule)
    benchmark_code = ColorCode(d=d, rounds=rounds, cnot_schedule=benchmark_dict, p_cnot=p_cnot)
    
    # Build custom code with different schedules for bulk vs edge plaquettes
    custom_dict = schedule_to_cnot_dict_by_color_and_type(colorcode_init, custom_schedules_by_color_and_type)
    custom_code = ColorCode(d=d, rounds=rounds, cnot_schedule=custom_dict, p_cnot=p_cnot)
    
    # Find minimal errors
    print("Finding minimal undetectable logical errors...")
    
    benchmark_errors = benchmark_code.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False
    )
    print(f"Benchmark: found error of weight {len(benchmark_errors)}")
    
    custom_errors = custom_code.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False
    )
    print(f"Custom: found error of weight {len(custom_errors)}")
    
    # Visualize benchmark - all errors combined
    fig1, ax1 = visualize_minimal_error(benchmark_code, benchmark_errors, 
                                        f"\nBenchmark Schedule")
    plt.savefig('minimal_error_benchmark.png', dpi=150)
    plt.show()
    
    # Visualize benchmark - each error separately in a grid
    print("\nVisualizing benchmark errors in grid...")
    visualize_errors_grid(benchmark_code, benchmark_errors, 
                         title_prefix="Benchmark Schedule",
                         save_name="benchmark_errors_grid.png")
    
    # Visualize custom - all errors combined
    fig2, ax2 = visualize_minimal_error(custom_code, custom_errors,
                                        f"\nCustom Per-Color Schedule")
    plt.savefig('minimal_error_custom.png', dpi=150)
    plt.show()
    
    # Visualize custom - each error separately in a grid
    print("\nVisualizing custom errors in grid...")
    visualize_errors_grid(custom_code, custom_errors,
                         title_prefix="Custom Per-Color Schedule", 
                         save_name="custom_errors_grid.png")
    
    # Print detailed error info with minimal representatives
    print("\n" + "="*60)
    print("BENCHMARK MINIMAL ERROR - Simplest Representatives")
    print("="*60)
    print_minimal_representatives(benchmark_errors)
    
    print("\n" + "="*60)
    print("CUSTOM MINIMAL ERROR - Simplest Representatives")
    print("="*60)
    print_minimal_representatives(custom_errors)
    
    return benchmark_errors, custom_errors


def print_minimal_representatives(errors):
    """Print the minimal representative for each error component."""
    total_qubits = set()
    
    for error_idx, error in enumerate(errors):
        error_str = str(error)
        locations = error_str.split('CircuitErrorLocation {')[1:]
        
        # Parse all locations
        parsed_locations = []
        for loc_idx, loc in enumerate(locations):
            match = re.search(r'flipped_pauli_product:\s*([^\n]+)', loc)
            if match:
                pauli_product = match.group(1).strip()
                paulis = parse_pauli_product(pauli_product)
                parsed_locations.append({
                    'loc_idx': loc_idx,
                    'pauli_product': pauli_product,
                    'paulis': paulis,
                    'num_qubits': len(paulis)
                })
        
        if parsed_locations:
            # Find minimum
            min_loc = min(parsed_locations, key=lambda x: x['num_qubits'])
            
            print(f"\nError {error_idx + 1}: {min_loc['pauli_product']}")
            print(f"  Qubits affected: {min_loc['num_qubits']}")
            print(f"  (Selected from {len(parsed_locations)} possible representations)")
            
            for pauli, qubit_idx, x, y in min_loc['paulis']:
                print(f"    {pauli} on qubit {qubit_idx} at ({x}, {y})")
                total_qubits.add((x, y))
    
    print(f"\nTotal unique qubit locations: {len(total_qubits)}")
    print(f"Qubit positions: {sorted(total_qubits)}")


if __name__ == "__main__":
    benchmark_errors, custom_errors = compare_minimal_errors(d=11, rounds=1, p_cnot=1e-3)

