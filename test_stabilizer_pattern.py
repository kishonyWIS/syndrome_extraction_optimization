import numpy as np
from rotated_surface_code import RotatedSurfaceCode

def visualize_stabilizers(d=5):
    """
    Visualize the stabilizer pattern for a d=5 rotated surface code.
    """
    code = RotatedSurfaceCode(d)
    
    print(f"Rotated Surface Code with d={d}")
    print(f"Data qubits: {code.n_data_qubits}")
    print(f"X stabilizers: {code.n_x_stabilizers}")
    print(f"Z stabilizers: {code.n_z_stabilizers}")
    print()
    
    print('h_x:')
    print(code.h_x)
    print('h_z:')
    print(code.h_z)

    print('l_x:')
    print(code.l_x)
    print('l_z:')
    print(code.l_z)

if __name__ == "__main__":
    visualize_stabilizers(5) 