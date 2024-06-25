import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

# Parameters
N = 5  # Number of variables
original_mean = np.full(N, 0.5)
covariance = np.diag(np.full(N, 0.1))
original_a = np.array([0, 0.1, 0.2, 0.3, 0.4])  # Lower truncation bounds for each variable
original_b = np.array([1, 0.9, 0.8, 0.7, 0.6])  # Upper truncation bounds for each variable
num_samples = 1000

# Fix one variable, let's say the first one
fixed_variable_index = 0
fixed_mean = original_mean[fixed_variable_index]

# Calculate the shift vector R
R = np.zeros(N)
shift_value = (fixed_mean - np.mean(original_mean)) / (N - 1)
for i in range(N):
    if i != fixed_variable_index:
        R[i] = shift_value

# Adjust means and truncation bounds of other variables by adding the shift vector R
adjusted_mean = original_mean + R
adjusted_a = original_a + R
adjusted_b = original_b + R

# Function to sample from a truncated normal distribution
def truncated_multivariate_normal(mean, cov, a, b, size):
    dimension = len(mean)
    samples = np.zeros((size, dimension))
    
    for i in range(dimension):
        a_i = (a[i] - mean[i]) / np.sqrt(cov[i, i])
        b_i = (b[i] - mean[i]) / np.sqrt(cov[i, i])
        samples[:, i] = truncnorm.rvs(a_i, b_i, loc=mean[i], scale=np.sqrt(cov[i, i]), size=size)
    
    return samples

# Generate samples with adjusted means and truncation bounds
samples = truncated_multivariate_normal(adjusted_mean, covariance, adjusted_a, adjusted_b, num_samples)

# Check which variable is the maximum for each sample
max_counts = np.zeros(N)
for sample in samples:
    max_counts[np.argmax(sample)] += 1

# Plotting the distribution of maximum occurrences
plt.bar(range(N), max_counts, alpha=0.7)
plt.xlabel('Variable Index')
plt.ylabel('Counts of Being Maximum')
plt.title('Distribution of Maximum Occurrences for Each Variable')
plt.grid(True)
plt.show()

print("Shift vector R:", R)
print("Adjusted mean vector:", adjusted_mean)
print("Adjusted lower truncation bounds:", adjusted_a)
print("Adjusted upper truncation bounds:", adjusted_b)
print("Counts of being maximum for each variable:", max_counts)
