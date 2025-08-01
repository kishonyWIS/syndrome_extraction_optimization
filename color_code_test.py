from color_code_stim import ColorCode
import numpy as np

colorcode = ColorCode(
    d=7,
    rounds=7,
    cnot_schedule="tri_optimal",  # Default CNOT schedule optimized in our paper.
    p_cnot=1e-3,
)

alpha = 0.01
num_fails, info = colorcode.simulate(
    1000000,  # Number of shots
    full_output=True,  # Whether to get additional information on decoding outputs
    alpha=alpha,  # Significance level of the confidence interval of pfail
    verbose=True,
    seed=42,
)

print(num_fails)
print(info)



# sample from circuit and decode and calculate error rate

circuit = colorcode.circuit
dem = circuit.detector_error_model()
sampler = dem.compile_sampler()

# Sample with error tracking
dets, obs, errs = sampler.sample(shots=1000000, return_errors=True)    
# Decode each shot
predicted_observables = colorcode.decode(dets)

# Handle the return type - decode returns either ndarray or Tuple[ndarray, dict]
if isinstance(predicted_observables, tuple):
    predicted_observables = predicted_observables[0]  # Extract the array from the tuple

# Ensure predicted_observables has the right shape for comparison
if predicted_observables.ndim == 1:
    predicted_observables = predicted_observables.reshape(-1, 1)

# Check if any logical operator was wrongly corrected
# For multiple observables, check if any observable was wrongly corrected
logical_errors = np.logical_xor(predicted_observables, obs).any(axis=1)  # Any observable wrong
logical_error_rate = float(np.mean(logical_errors))
print(f"Logical error rate (after decoding): {logical_error_rate}")







# # Second test: using dict of list of lists for cnot_schedule (equivalent to 'tri_optimal')
# # For d=5, the number of Z and X stabilizers is determined by the code construction.
# # We'll use the same schedule for each stabilizer as in 'tri_optimal'.

# # 'tri_optimal' schedule as per the code: [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
# tri_optimal_schedule = [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2]
# z_target_schedule = np.argsort(tri_optimal_schedule[:6]).tolist()
# x_target_schedule = np.argsort(tri_optimal_schedule[6:]).tolist()

# # To mimic the default, we need to know the number of Z and X stabilizers for d=5
# num_z_stabs = len(colorcode.qubit_groups['anc_Z'])
# num_x_stabs = len(colorcode.qubit_groups['anc_X'])

# cnot_schedule_dict = {
#     'Z': [z_target_schedule for _ in range(num_z_stabs)],
#     'X': [x_target_schedule for _ in range(num_x_stabs)],
# }

# colorcode_dict = ColorCode(
#     d=7,
#     rounds=7,
#     cnot_schedule=cnot_schedule_dict,
#     p_cnot=1e-3,
# )

# num_fails_dict, info_dict = colorcode_dict.simulate(
#     100000,
#     full_output=True,
#     alpha=alpha,
#     verbose=True,
#     seed=42,
# )

# print("With dict cnot_schedule (split):")
# print(num_fails_dict)
# print(info_dict)