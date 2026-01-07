"""
Visualize the custom per-color CNOT schedule for the color code.
"""

import numpy as np
import matplotlib.pyplot as plt
from color_code_stim import ColorCode
from color_code_cnot_optimization import draw_lattice_with_individual_colors


def schedule_to_cnot_dict_by_color(colorcode, schedules_by_color):
    """Convert per-color schedules to a per-stabilizer CNOT schedule dict."""
    cnot_orders = {}
    for color, schedule in schedules_by_color.items():
        z_order = np.argsort(schedule[:6]).tolist()
        x_order = np.argsort(schedule[6:]).tolist()
        cnot_orders[color] = {'Z': z_order, 'X': x_order}
    
    z_schedules = []
    x_schedules = []
    
    for anc_qubit in colorcode.qubit_groups['anc_Z']:
        color = anc_qubit['color']
        z_schedules.append(cnot_orders[color]['Z'])
    
    for anc_qubit in colorcode.qubit_groups['anc_X']:
        color = anc_qubit['color']
        x_schedules.append(cnot_orders[color]['X'])
    
    return {'Z': z_schedules, 'X': x_schedules}


def visualize_stabilizer_schedules_by_color(colorcode, stabilizer_type='X', colormap_name='viridis'):
    """
    Visualize the CNOT schedules for stabilizers, grouped by plaquette color.
    
    Args:
        colorcode: ColorCode instance
        stabilizer_type: 'X' or 'Z' stabilizers to visualize
        colormap_name: Name of matplotlib colormap to use
    """
    anc_qubits = colorcode.qubit_groups[f'anc_{stabilizer_type}']
    schedules = colorcode.cnot_schedule[stabilizer_type]
    tanner_graph = colorcode.tanner_graph

    offsets = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
    
    # Group stabilizers by color
    stabs_by_color = {'r': [], 'g': [], 'b': []}
    for stab_idx, anc_qubit in enumerate(anc_qubits):
        color = anc_qubit['color']
        stabs_by_color[color].append((stab_idx, anc_qubit))
    
    color_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
    
    for plaq_color, stabs in stabs_by_color.items():
        if not stabs:
            continue
            
        print(f"\n{'='*60}")
        print(f"{color_names[plaq_color]} plaquettes - {stabilizer_type} stabilizers")
        print(f"{'='*60}")
        
        for stab_idx, anc_qubit in stabs:
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
            
            print(f"  Stabilizer {stab_idx} at ({anc_qubit['x']}, {anc_qubit['y']}): order {schedule}")
            print(f"    Data qubits: {data_qubits_list}")
            
            # Create colormap for the qubits
            if data_qubits_list:
                qubit_colors = {}
                
                for i, qubit_name in enumerate(data_qubits_list):
                    if len(data_qubits_list) > 1:
                        color_value = i / (len(data_qubits_list) - 1)
                    else:
                        color_value = 0.5
                    
                    color = plt.get_cmap(colormap_name)(color_value)
                    qubit_colors[qubit_name] = color
                
                # Draw with individual colors
                ax = draw_lattice_with_individual_colors(colorcode, qubit_colors=qubit_colors)
                plt.title(f"{color_names[plaq_color]} {stabilizer_type}-stabilizer {stab_idx}\nCNOT Order: {schedule}")
                plt.show()


def visualize_all_schedules_overview(colorcode, stabilizer_type='Z'):
    """
    Create an overview showing all stabilizer schedules at once with color coding.
    """
    anc_qubits = colorcode.qubit_groups[f'anc_{stabilizer_type}']
    schedules = colorcode.cnot_schedule[stabilizer_type]
    tanner_graph = colorcode.tanner_graph

    offsets = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
    
    # Create a figure showing the first CNOT target for each stabilizer
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for cnot_step in range(6):
        ax = axes[cnot_step]
        
        # Draw the base lattice
        ax = colorcode.draw_lattice(ax=ax, show_data_qubits=True)
        
        # Highlight the data qubit targeted at this step for each stabilizer
        qubit_colors = {}
        plaq_color_map = {'r': 'red', 'g': 'green', 'b': 'blue'}
        
        for stab_idx, anc_qubit in enumerate(anc_qubits):
            schedule = schedules[stab_idx]
            plaq_color = anc_qubit['color']
            
            if cnot_step < len(schedule):
                offset_idx = schedule[cnot_step]
                offset = offsets[offset_idx % 6]
                data_qubit_x = anc_qubit["x"] + offset[0]
                data_qubit_y = anc_qubit["y"] + offset[1]
                data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                
                try:
                    data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                    qubit_colors[data_qubit_name] = plaq_color_map[plaq_color]
                except ValueError:
                    continue
        
        # Redraw with highlighted qubits
        for qubit_name, color in qubit_colors.items():
            try:
                qubit = tanner_graph.vs.find(name=qubit_name)
                ax.scatter([qubit['x']], [qubit['y']], c=color, s=200, 
                          edgecolors='black', linewidths=2, zorder=10, marker='o')
            except:
                pass
        
        ax.set_title(f'CNOT Step {cnot_step + 1}')
    
    plt.suptitle(f'{stabilizer_type}-stabilizer CNOT Schedule Overview\n(Colors show which plaquette color targets each data qubit)', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('schedule_overview.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    d = 7
    rounds = 2
    p_cnot = 1e-3
    
    # Define custom per-color schedules
    custom_schedules_by_color = {
        'r': [1, 4, 2, 6, 3, 5],
        'b': [1, 5, 3, 6, 2, 4],
        'g': [4, 2, 6, 3, 5, 1],
    }
    for color in ['r', 'g', 'b']:
        custom_schedules_by_color[color] = custom_schedules_by_color[color] + [x + 6 for x in custom_schedules_by_color[color]]
    
    print("Custom per-color schedules (raw):")
    for color, sched in custom_schedules_by_color.items():
        print(f"  {color}: {sched}")
    
    print("\nCustom per-color CNOT orders (after argsort):")
    for color, sched in custom_schedules_by_color.items():
        z_order = np.argsort(sched[:6]).tolist()
        x_order = np.argsort(sched[6:]).tolist()
        print(f"  {color.upper()}: Z order: {z_order}, X order: {x_order}")
    
    # Build initial code to get stabilizer info
    colorcode_init = ColorCode(d=d, rounds=rounds, cnot_schedule="tri_optimal", p_cnot=p_cnot)
    
    # Convert to per-stabilizer schedule
    cnot_schedule_dict = schedule_to_cnot_dict_by_color(colorcode_init, custom_schedules_by_color)
    
    # Build the color code with custom schedule
    colorcode = ColorCode(
        d=d,
        rounds=rounds,
        cnot_schedule=cnot_schedule_dict,
        p_cnot=p_cnot,
    )
    
    # Visualize overview of all schedules
    print("\n" + "="*60)
    print("Visualizing schedule overview...")
    print("="*60)
    visualize_all_schedules_overview(colorcode, stabilizer_type='Z')
    
    # Visualize individual stabilizers by color
    # Uncomment below to see individual stabilizer plots:
    # visualize_stabilizer_schedules_by_color(colorcode, stabilizer_type='Z', colormap_name='viridis')

