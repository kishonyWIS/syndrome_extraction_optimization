import numpy as np

class CSSCode:
    def __init__(self, h_x, h_z, l_x, l_z):
        """
        h_x: numpy array (n_x_stabilizers, n_data_qubits) - X stabilizer parity check matrix
        h_z: numpy array (n_z_stabilizers, n_data_qubits) - Z stabilizer parity check matrix
        l_x: numpy array (n_logicals, n_data_qubits) - X logical operators
        l_z: numpy array (n_logicals, n_data_qubits) - Z logical operators
        """
        self.h_x = h_x
        self.h_z = h_z
        self.l_x = l_x
        self.l_z = l_z
        self.n_data_qubits = h_x.shape[1]
        self.n_x_stabilizers = h_x.shape[0]
        self.n_z_stabilizers = h_z.shape[0]
        # Assign integer indices
        self.data_qubit_indices = list(range(self.n_data_qubits))
        self.x_ancilla_indices = list(range(self.n_data_qubits, self.n_data_qubits + self.n_x_stabilizers))
        self.z_ancilla_indices = list(range(self.n_data_qubits + self.n_x_stabilizers,
                                            self.n_data_qubits + self.n_x_stabilizers + self.n_z_stabilizers))
        # Store mutable CX orders for each stabilizer
        self.cx_orders_x = [list(np.flatnonzero(h_x[i])) for i in range(self.n_x_stabilizers)]
        self.cx_orders_z = [list(np.flatnonzero(h_z[i])) for i in range(self.n_z_stabilizers)]

    def get_cx_order(self, stabilizer_type, stabilizer_index):
        """
        Returns the ordered list of data qubit indices for the given stabilizer.
        stabilizer_type: 'X' or 'Z'
        stabilizer_index: int
        """
        if stabilizer_type == 'X':
            return self.cx_orders_x[stabilizer_index]
        elif stabilizer_type == 'Z':
            return self.cx_orders_z[stabilizer_index]
        else:
            raise ValueError('stabilizer_type must be "X" or "Z"')

    def set_cx_order(self, stabilizer_type, stabilizer_index, new_order):
        """
        Sets a new ordered list of data qubit indices for the given stabilizer.
        new_order must be a permutation of the original data qubits for that stabilizer.
        """
        if stabilizer_type == 'X':
            orig = set(np.flatnonzero(self.h_x[stabilizer_index]))
            if set(new_order) != orig:
                raise ValueError('new_order must be a permutation of the original data qubits for this stabilizer')
            self.cx_orders_x[stabilizer_index] = list(new_order)
        elif stabilizer_type == 'Z':
            orig = set(np.flatnonzero(self.h_z[stabilizer_index]))
            if set(new_order) != orig:
                raise ValueError('new_order must be a permutation of the original data qubits for this stabilizer')
            self.cx_orders_z[stabilizer_index] = list(new_order)
        else:
            raise ValueError('stabilizer_type must be "X" or "Z"')

    def get_ancilla(self, stabilizer_type, stabilizer_index):
        """
        Returns the integer ancilla index for the given stabilizer.
        """
        if stabilizer_type == 'X':
            return self.x_ancilla_indices[stabilizer_index]
        elif stabilizer_type == 'Z':
            return self.z_ancilla_indices[stabilizer_index]
        else:
            raise ValueError('stabilizer_type must be "X" or "Z"')

    def ancilla_index_to_stabilizer(self, ancilla_index):
        """
        Given an ancilla qubit index, returns a tuple (stabilizer_type, stabilizer_index),
        where stabilizer_type is 'X' or 'Z' and stabilizer_index is the index of the stabilizer.
        Raises ValueError if the index is not an ancilla.
        """
        if ancilla_index in self.x_ancilla_indices:
            return ('X', self.x_ancilla_indices.index(ancilla_index))
        elif ancilla_index in self.z_ancilla_indices:
            return ('Z', self.z_ancilla_indices.index(ancilla_index))
        else:
            raise ValueError(f"Index {ancilla_index} is not an ancilla qubit.") 