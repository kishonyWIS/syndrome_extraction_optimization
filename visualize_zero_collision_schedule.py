"""
Visualize zero-collision schedules from the CSV file.
Shows the CNOT order numbers around each bulk plaquette.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from color_code_stim import ColorCode
import csv
import ast


def load_schedule_from_csv(csv_file: str, index: int = 1):
    """
    Load a specific schedule from the zero_collision_schedules.csv file.
    
    Args:
        csv_file: Path to the CSV file
        index: 1-based index of the schedule to load
        
    Returns:
        Dictionary with schedule info
    """
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['index']) == index:
                return {
                    'index': int(row['index']),
                    'r_schedule': ast.literal_eval(row['r_schedule']),
                    'g_schedule': ast.literal_eval(row['g_schedule']),
                    'b_schedule': ast.literal_eval(row['b_schedule']),
                    'orig_r_schedule': ast.literal_eval(row['orig_r_schedule']),
                    'orig_g_schedule': ast.literal_eval(row['orig_g_schedule']),
                    'orig_b_schedule': ast.literal_eval(row['orig_b_schedule']),
                    'circuit_distance': int(row['circuit_distance']),
                    'collisions': int(row['collisions']),
                }
    raise ValueError(f"Schedule with index {index} not found")


def visualize_schedule(schedule_info: dict, d: int = 7, save_path: str = None, 
                       show_numbers: bool = True):
    """
    Visualize a zero-collision schedule showing CNOT order numbers.
    
    Args:
        schedule_info: Dictionary with r_schedule, g_schedule, b_schedule
        d: Code distance
        save_path: Optional path to save the figure
        show_numbers: Whether to show CNOT schedule numbers on plaquettes
        title: Custom title (if None, auto-generated)
    """
    # Build a color code to get the lattice structure
    colorcode = ColorCode(d=d, rounds=1, cnot_schedule="tri_optimal", p_cnot=0)
    
    # Get qubit groups
    data_qubits = colorcode.qubit_groups['data']
    z_ancillas = colorcode.qubit_groups['anc_Z']
    
    # Canonical offsets for the 6 data qubits around a plaquette
    # Order: upper-left, upper-right, right, lower-right, lower-left, left
    CANONICAL_OFFSETS = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
    
    # Color mapping
    plaq_colors = {
        'r': '#ffcccc',  # Light red
        'g': '#ccffcc',  # Light green  
        'b': '#ccccff',  # Light blue
    }
    text_colors = {
        'r': '#cc0000',  # Dark red
        'g': '#008800',  # Dark green
        'b': '#0000cc',  # Dark blue
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get schedules by color
    schedules = {
        'r': schedule_info['r_schedule'],
        'g': schedule_info['g_schedule'],
        'b': schedule_info['b_schedule'],
    }
    
    # Collect all positions to set axis limits
    all_x = []
    all_y = []
    
    # Y-axis stretch factor for regular hexagons
    # Original coords: width=8, height=2, ratio=4:1
    # Regular hexagon: width:height = 2:sqrt(3)
    # So y_scale = 2*sqrt(3) ≈ 3.464
    y_scale = 2 * np.sqrt(3)
    
    # Draw data qubits first (as small dots)
    for dq in data_qubits:
        ax.scatter(dq['x'], dq['y'] * y_scale, c='white', s=80, edgecolors='#666666', 
                  linewidths=1, zorder=5)
        all_x.append(dq['x'])
        all_y.append(dq['y'] * y_scale)
    
    # Create a set of data qubit positions for quick lookup
    data_pos = {(dq['x'], dq['y']) for dq in data_qubits}
    
    # Draw plaquettes as polygons connecting data qubits
    for anc in z_ancillas:
        all_x.append(anc['x'])
        all_y.append(anc['y'] * y_scale)
        
        color = anc['color']
        
        # Get the data qubits around this plaquette in order
        data_positions = []
        for dx, dy in CANONICAL_OFFSETS:
            pos = (anc['x'] + dx, anc['y'] + dy)
            if pos in data_pos:
                # Scale y coordinate
                data_positions.append((pos[0], pos[1] * y_scale))
        
        num_neighbors = len(data_positions)
        
        if num_neighbors >= 3:
            # Draw polygon connecting the data qubits
            poly = Polygon(data_positions, 
                          facecolor=plaq_colors[color],
                          edgecolor='#557799',
                          linewidth=1.5,
                          alpha=0.6,
                          zorder=1)
            ax.add_patch(poly)
        
        # Draw ancilla qubit (small dark dot)
        ax.scatter(anc['x'], anc['y'] * y_scale, c='#333333', s=25, zorder=10)
        
        # Add schedule numbers for all plaquettes (only for existing data qubits)
        if show_numbers:
            schedule = schedules[color]
            
            for i, (dx, dy) in enumerate(CANONICAL_OFFSETS):
                pos = (anc['x'] + dx, anc['y'] + dy)
                if pos in data_pos:
                    # Position the number between ancilla and data qubit
                    num_x = anc['x'] + dx * 0.7
                    num_y = (anc['y'] + dy * 0.7) * y_scale
                    
                    ax.text(num_x, num_y, str(schedule[i]), 
                           fontsize=14, fontweight='bold',
                           color=text_colors[color],
                           ha='center', va='center',
                           zorder=15)
    
    # Set axis properties
    margin = 3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    import os
    
    # Load the first schedule
    csv_file = 'results/zero_collision_schedules.csv'
    schedule_info = load_schedule_from_csv(csv_file, index=1)
    
    print(f"Loaded schedule #{schedule_info['index']}:")
    print(f"  R: {schedule_info['r_schedule']}")
    print(f"  G: {schedule_info['g_schedule']}")
    print(f"  B: {schedule_info['b_schedule']}")
    print(f"  Circuit distance: {schedule_info['circuit_distance']}")
    print(f"  Collisions: {schedule_info['collisions']}")
    
    # Create output directory
    os.makedirs('results/schedule_visualizations', exist_ok=True)
    
    d = 9
    
    # Visualize with numbers
    print("\nGenerating visualization with schedule numbers...")
    visualize_schedule(
        schedule_info,
        d=d,
        save_path='results/schedule_visualizations/zero_collision_01_with_numbers.pdf',
        show_numbers=True
    )
    
    # Visualize without numbers
    print("\nGenerating visualization without schedule numbers...")
    visualize_schedule(
        schedule_info, 
        d=d,
        save_path='results/schedule_visualizations/zero_collision_01_lattice_only.pdf',
        show_numbers=False
    )
