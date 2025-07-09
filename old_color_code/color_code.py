import numpy as np
from css_code import CSSCode

class ColorCode(CSSCode):
    def __init__(self, d: int):
        """
        Initialize a color code on a triangular patch of the honeycomb lattice with code distance d.
        Args:
            d: Code distance (number of hexagons along an edge, d >= 3, odd)
        """
        if d < 3 or d % 2 == 0:
            raise ValueError("Distance d must be odd and at least 3")
        self.d = d
        self.row_lengths = self._get_row_lengths(d)
        self.num_rows = len(self.row_lengths)
        self.vertex_coords, self.coord_to_index = self._generate_vertex_map()
        self.plaquettes = self._generate_plaquettes()
        h_x, h_z = self._build_stabilizer_matrices()
        l_x, l_z = self._build_logical_operators()
        super().__init__(h_x, h_z, l_x, l_z)

    def _get_row_lengths(self, d):
        if d == 3:
            return [3, 2, 1, 1]
        else:
            prev = self._get_row_lengths(d - 2)
            return [d, d - 1, d - 2] + prev

    def _generate_vertex_map(self):
        vertex_coords = []
        coord_to_index = {}
        idx = 0
        for row, n_cols in enumerate(self.row_lengths):
            for col in range(n_cols):
                vertex_coords.append((row, col))
                coord_to_index[(row, col)] = idx
                idx += 1
        return vertex_coords, coord_to_index

    def _generate_plaquettes(self):
        plaquettes = []
        num_rows = self.num_rows
        row_lengths = self.row_lengths
        for row in range(num_rows):
            n_max = (row_lengths[row] - 1) // 2 if row % 3 != 0 else (row_lengths[row] - 2) // 2
            for n in range(n_max + 1):
                if row % 3 == 2:
                    candidates = [
                        (row, 2 * n), (row, 2 * n + 1),
                        (row + 1, 2 * n), (row + 1, 2 * n + 1),
                        (row - 1, 2 * n), (row - 1, 2 * n + 1)
                    ]
                elif row % 3 == 1:
                    candidates = [
                        (row, 2 * n), (row, 2 * n - 1),
                        (row + 1, 2 * n), (row + 1, 2 * n - 1),
                        (row - 1, 2 * n), (row - 1, 2 * n + 1)
                    ]
                else:  # row % 3 == 0
                    candidates = [
                        (row, 2 * n + 1), (row, 2 * n + 2),
                        (row + 1, 2 * n), (row + 1, 2 * n + 1),
                        (row - 1, 2 * n + 1), (row - 1, 2 * n + 2)
                    ]
                plaq = [v for v in candidates if v in self.coord_to_index]
                if len(plaq) >= 3:
                    plaquettes.append(plaq)
        return plaquettes

    def _build_stabilizer_matrices(self):
        n = len(self.vertex_coords)
        n_stab = len(self.plaquettes)
        h_x = np.zeros((n_stab, n), dtype=np.int8)
        h_z = np.zeros((n_stab, n), dtype=np.int8)
        for i, plaq in enumerate(self.plaquettes):
            for v in plaq:
                idx = self.coord_to_index[v]
                h_x[i, idx] = 1
                h_z[i, idx] = 1
        return h_x, h_z

    def _build_logical_operators(self):
        n = len(self.vertex_coords)
        l_x = np.zeros((2, n), dtype=np.int8)
        l_z = np.zeros((2, n), dtype=np.int8)
        # Logical operator 1: all qubits (row, col=0)
        for row in range(self.num_rows):
            if row % 3 in [0, 2] and (row, 0) in self.coord_to_index:
                idx = self.coord_to_index[(row, 0)]
                l_x[0, idx] = 1
                l_z[0, idx] = 1
        # Logical operator 2: all qubits (row=0, col)
        for col in range(self.row_lengths[0]):
            if (0, col) in self.coord_to_index:
                idx = self.coord_to_index[(0, col)]
                l_x[1, idx] = 1
                l_z[1, idx] = 1
        return l_x, l_z

    def get_qubit_coordinates(self, qubit_idx: int) -> tuple[int, int]:
        return self.vertex_coords[qubit_idx]

    def get_qubit_index(self, row: int, col: int) -> int:
        return self.coord_to_index[(row, col)] 