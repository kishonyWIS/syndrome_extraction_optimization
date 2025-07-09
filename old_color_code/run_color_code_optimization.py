from stim_error_analyzer import optimize_circuit_cx_orders
from color_code import ColorCode
import matplotlib.pyplot as plt


def main():
    d = 9
    n_steps = 100
    n_rounds = 3  # Number of rounds for the memory experiment
    css_code = ColorCode(d)
    print(f"Running optimization for Color Code with distance {d} for {n_steps} steps, n_rounds={n_rounds}...")
    results = optimize_circuit_cx_orders(css_code, n_shots=10000, noise_prob=0.01, n_steps=n_steps, n_rounds=n_rounds)
    if results is not None:
        steps = list(range(1, len(results) + 1))
        logical_error_rates = [r[0] for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, logical_error_rates, marker='o')
        plt.xlabel("Optimization Step")
        plt.ylabel("Logical Error Rate")
        plt.title("Logical Error Rate vs Optimization Step")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Optimization did not complete successfully.")

if __name__ == "__main__":
    main() 