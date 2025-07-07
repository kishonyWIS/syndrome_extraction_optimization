import stim
import numpy as np

def build_memory_experiment_circuit(css_code, n_rounds, noise_prob, memory_type='Z'):
    """
    Constructs a stim.Circuit for a memory experiment with the given CSS code, including DETECTOR and OBSERVABLE_INCLUDE instructions.
    Args:
        css_code: CSSCode object
        n_rounds: int, number of noisy rounds (total rounds = n_rounds + 2)
        noise_prob: float, probability for DEPOLARIZE2
        memory_type: 'Z' or 'X' (Z memory or X memory experiment)
    Returns:
        stim.Circuit object
    """
    circuit = stim.Circuit()
    total_rounds = n_rounds + 2
    noisy_rounds = set(range(1, n_rounds + 1))  # Only rounds 1..n_rounds are noisy
    n_data = css_code.n_data_qubits

    # --- Initialize data qubits ---
    if memory_type == 'Z':
        circuit.append('R', list(range(n_data)))
    elif memory_type == 'X':
        circuit.append('RX', list(range(n_data)))
    else:
        raise ValueError("memory_type must be 'Z' or 'X'")

    # Track measurement indices for each stabilizer's ancilla in each round
    z_meas_indices = [[] for _ in range(css_code.n_z_stabilizers)]
    x_meas_indices = [[] for _ in range(css_code.n_x_stabilizers)]
    current_meas_index = 0

    for round_idx in range(total_rounds):
        noisy = round_idx in noisy_rounds
        # Z stabilizers
        for z_idx in range(css_code.n_z_stabilizers):
            anc = css_code.get_ancilla('Z', z_idx)
            circuit.append('R', [anc])
            for dq in css_code.get_cx_order('Z', z_idx):
                if noisy:
                    circuit.append('DEPOLARIZE2', [anc, dq], noise_prob)
                circuit.append('CX', [dq, anc])
            circuit.append('M', [anc])
            z_meas_indices[z_idx].append(current_meas_index)
            current_meas_index += 1
        # X stabilizers
        for x_idx in range(css_code.n_x_stabilizers):
            anc = css_code.get_ancilla('X', x_idx)
            circuit.append('RX', [anc])
            for dq in css_code.get_cx_order('X', x_idx):
                if noisy:
                    circuit.append('DEPOLARIZE2', [anc, dq], noise_prob)
                circuit.append('CX', [anc, dq])
            circuit.append('MX', [anc])
            x_meas_indices[x_idx].append(current_meas_index)
            current_meas_index += 1

    # --- Final data qubit measurements ---
    data_meas_indices = []
    if memory_type == 'Z':
        circuit.append('M', list(range(n_data)))
        data_meas_indices = list(range(current_meas_index, current_meas_index + n_data))
        current_meas_index += n_data
    elif memory_type == 'X':
        circuit.append('MX', list(range(n_data)))
        data_meas_indices = list(range(current_meas_index, current_meas_index + n_data))
        current_meas_index += n_data

    # Use relative indices for DETECTOR instructions
    total_meas = current_meas_index
    # Add DETECTOR instructions for Z stabilizers
    for z_idx in range(css_code.n_z_stabilizers):
        for r in range(1, total_rounds):
            idx1 = z_meas_indices[z_idx][r-1] - total_meas
            idx2 = z_meas_indices[z_idx][r] - total_meas
            circuit.append('DETECTOR', [stim.target_rec(idx1), stim.target_rec(idx2)])
    # Add DETECTOR instructions for X stabilizers
    for x_idx in range(css_code.n_x_stabilizers):
        for r in range(1, total_rounds):
            idx1 = x_meas_indices[x_idx][r-1] - total_meas
            idx2 = x_meas_indices[x_idx][r] - total_meas
            circuit.append('DETECTOR', [stim.target_rec(idx1), stim.target_rec(idx2)])

    # --- OBSERVABLE_INCLUDE for logicals ---
    if memory_type == 'Z':
        for log_idx, log_row in enumerate(css_code.l_z):
            qubits = np.flatnonzero(log_row)
            obs_targets = [stim.target_rec(data_meas_indices[q] - total_meas) for q in qubits]
            circuit.append('OBSERVABLE_INCLUDE', obs_targets, log_idx)
    elif memory_type == 'X':
        for log_idx, log_row in enumerate(css_code.l_x):
            qubits = np.flatnonzero(log_row)
            obs_targets = [stim.target_rec(data_meas_indices[q] - total_meas) for q in qubits]
            circuit.append('OBSERVABLE_INCLUDE', obs_targets, log_idx)

    return circuit 