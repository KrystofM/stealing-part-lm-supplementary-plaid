import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Function to generate truncated normal samples
def truncated_normal(mean, sigma, lower, upper, size=1):
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    truncated_distribution = stats.truncnorm(a, b, loc=mean, scale=sigma)
    samples = truncated_distribution.rvs(size=size)
    return samples

# Define parameters for N variables
N = 3  # Number of variables
means = np.linspace(0.1, 1, N)
sigmas = np.full(N, 0.1)
lowers = np.linspace(0, 0.9, N)
uppers = lowers + 0.1
sizes = np.full(N, 1000)

# Generate samples for each variable
variables = {f'Var{i}': truncated_normal(means[i], sigmas[i], lowers[i], uppers[i], sizes[i]) for i in range(N)}

# Function to calculate probabilities
def calculate_probabilities(variables, shifts):
    shifted_vars = {k: v + shifts.get(k, 0) for k, v in variables.items()}
    results = {}
    for k1 in shifted_vars:
        for k2 in shifted_vars:
            if k1 != k2:
                results[f'P({k1} > {k2})'] = np.mean(shifted_vars[k1] > shifted_vars[k2])
    return results

# Optimization to find best shifts
best_shifts = {k: 0 for k in variables}  # Initialize shifts
best_results = calculate_probabilities(variables, best_shifts)
tolerance = 0.01
max_iterations = 100
iteration = 0

while iteration < max_iterations:
    iteration += 1
    for var in variables:
        current_shift = best_shifts[var]
        test_shifts = np.linspace(current_shift - 0.05, current_shift + 0.05, 5)
        for shift in test_shifts:
            temp_shifts = best_shifts.copy()
            temp_shifts[var] = shift
            temp_results = calculate_probabilities(variables, temp_shifts)
            max_diff = max(temp_results.values()) - min(temp_results.values())
            if max_diff < tolerance:
                best_shifts[var] = shift
                best_results = temp_results
                tolerance = max_diff  # Update tolerance to the new best found
# Print the best shifts and results
print("Best Shifts:")
print(best_shifts)
print("\nBest Results:")
print(best_results)
# Plotting results
fig, axs = plt.subplots(10, 10, figsize=(20, 20))  # Adjust subplot grid as needed
axs = axs.flatten()
for i, (key, val) in enumerate(variables.items()):
    axs[i].hist(val + best_shifts[key], bins=30, alpha=0.75, color=np.random.rand(3,), edgecolor='black')
    axs[i].set_title(f'Shifted {key}')
    axs[i].set_xlabel(f'{key} values')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()