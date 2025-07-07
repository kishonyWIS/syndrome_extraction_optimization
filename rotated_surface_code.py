import numpy as np
from css_code import CSSCode

class RotatedSurfaceCode(CSSCode):
    def __init__(self, d: int):
        """
        Initialize a rotated surface code with distance d.
        The code has d^2 data qubits arranged in a d x d grid.
        
        Args:
            d: The code distance (must be odd)
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be odd")
            
        self.d = d
        
        # Build stabilizer matrices
        h_x, h_z = self._build_stabilizer_matrices()
        l_x, l_z = self._build_logical_operators()
        
        super().__init__(h_x, h_z, l_x, l_z)
        
    def _build_stabilizer_matrices(self):
        """
        Builds X and Z stabilizer matrices.
        Returns (h_x, h_z) matrices.
        """
        d = self.d
        n_qubits = d * d
        
        def rc_to_idx(row: int, col: int) -> int:
            return row * d + col
        
        x_stabilizers = []
        z_stabilizers = []
        
        # Bulk stabilizers (2x2 blocks)
        for row in range(d - 1):
            for col in range(d - 1):
                # Checkerboard pattern: (row + col) % 2 == 0 for X, == 1 for Z
                if (row + col) % 2 == 0:
                    # X stabilizer
                    stab = np.zeros(n_qubits, dtype=np.int8)
                    stab[rc_to_idx(row, col)] = 1
                    stab[rc_to_idx(row, col+1)] = 1
                    stab[rc_to_idx(row+1, col)] = 1
                    stab[rc_to_idx(row+1, col+1)] = 1
                    x_stabilizers.append(stab)
                else:
                    # Z stabilizer
                    stab = np.zeros(n_qubits, dtype=np.int8)
                    stab[rc_to_idx(row, col)] = 1
                    stab[rc_to_idx(row, col+1)] = 1
                    stab[rc_to_idx(row+1, col)] = 1
                    stab[rc_to_idx(row+1, col+1)] = 1
                    z_stabilizers.append(stab)
        
        # Edge stabilizers (weight-2)
        # Left/right columns for X, top/bottom rows for Z
        for row in range(1, d, 2):
            # Left edge X
            stab = np.zeros(n_qubits, dtype=np.int8)
            stab[rc_to_idx(row, 0)] = 1
            stab[rc_to_idx(row+1, 0)] = 1
            x_stabilizers.append(stab)
            # Right edge X
            stab = np.zeros(n_qubits, dtype=np.int8)
            stab[rc_to_idx(row-1, d-1)] = 1
            stab[rc_to_idx(row, d-1)] = 1
            x_stabilizers.append(stab)
        for col in range(1, d, 2):
            # Top edge Z
            stab = np.zeros(n_qubits, dtype=np.int8)
            stab[rc_to_idx(0, col-1)] = 1
            stab[rc_to_idx(0, col)] = 1
            z_stabilizers.append(stab)
            # Bottom edge Z
            stab = np.zeros(n_qubits, dtype=np.int8)
            stab[rc_to_idx(d-1, col)] = 1
            stab[rc_to_idx(d-1, col+1)] = 1
            z_stabilizers.append(stab)
        
        h_x = np.array(x_stabilizers) if x_stabilizers else np.zeros((0, n_qubits), dtype=np.int8)
        h_z = np.array(z_stabilizers) if z_stabilizers else np.zeros((0, n_qubits), dtype=np.int8)
        return h_x, h_z
        
    def _build_logical_operators(self):
        """
        Builds logical X and Z operators.
        X operator runs across the top row.
        Z operator runs down the left column.
        Returns (l_x, l_z) matrices.
        """
        d = self.d
        n_qubits = d * d
        
        # Single X and Z logical operator
        l_x = np.zeros((1, n_qubits), dtype=np.int8)
        l_z = np.zeros((1, n_qubits), dtype=np.int8)
        
        # X logical = product of X on top row
        l_x[0, 0:d] = 1
        
        # Z logical = product of Z on leftmost column
        for i in range(d):
            l_z[0, i * d] = 1
            
        return l_x, l_z
        
    def get_qubit_coordinates(self, qubit_idx: int) -> tuple[int, int]:
        """
        Converts a qubit index to (row, col) coordinates.
        """
        row = qubit_idx // self.d
        col = qubit_idx % self.d
        return (row, col)
        
    def get_qubit_index(self, row: int, col: int) -> int:
        """
        Converts (row, col) coordinates to qubit index.
        """
        return row * self.d + col 