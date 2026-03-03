"""Quick script to compute circuit-level distance for d=11 parallel optimized circuit."""

import stim
from benchmark_circuits import build_parallel_circuit, load_zero_collision_schedule, apply_noise

# Load the zero-collision schedule
optimized_schedule = load_zero_collision_schedule('results/zero_collision_schedules.csv', index=0)
print('Loaded schedule:')
print(f'  R: {optimized_schedule["r_schedule"]}')
print(f'  G: {optimized_schedule["g_schedule"]}')
print(f'  B: {optimized_schedule["b_schedule"]}')

# Build d=11 parallel circuit (noiseless)
d = 13
rounds = 2
p_cnot = 0.001  # Need non-zero for error search

print(f'\nBuilding d={d} parallel optimized circuit...')
circuit = build_parallel_circuit(d, rounds, p_cnot, optimized_schedule)
# Apply noise so Stim's search has fault edges to explore (build_parallel_circuit returns noiseless circuit)
circuit = apply_noise(circuit, p_cnot, 'depolarize2_after_cnot')

print(f'Circuit has {circuit.num_qubits} qubits, {circuit.num_detectors} detectors')

# Find circuit-level distance
print(f'\nSearching for undetectable logical errors (this may take a while)...')

errors = circuit.search_for_undetectable_logical_errors(
    dont_explore_detection_event_sets_with_size_above=4,
    dont_explore_edges_with_degree_above=9999,
    dont_explore_edges_increasing_symptom_degree=False,
    canonicalize_circuit_errors=True
)

print(f'\nCircuit-level distance: {len(errors)}')
