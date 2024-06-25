import numpy as np
import scipy.integrate as integrate
from scipy.stats import truncnorm

def sample_uniform(l_i, h_i, n):
    # Sample n many samples from the uniform distribution
    samples = np.random.uniform(low=l_i, high=h_i, size=n)
    
    return samples

# Vectorized version
# P(P_i + r_i < P_n) = (1/n)^(1/(n-1)
def calculate_product_d_uniform(l_i, h_i, r_i, l_n, h_n):
    # Calculate the overlap of the intervals [l_i, h_i] and [l_n - r_i, h_n - r_i]
    effective_l_i = max(l_i, l_n - r_i)
    effective_h_i = min(h_i, h_n - r_i)
    
    if effective_l_i >= effective_h_i:
        return 0
    else:
        # The probability is the ratio of the overlap to the width of [l_n, h_n]
        return (effective_h_i - effective_l_i) / (h_n - l_n)
 
def find_r_i(l_i, h_i, l_n, h_n, mu, sigma, n):
    lower_bound = -10
    upper_bound = 10
    r_i = None
    min_diff = float('inf')
    min_r_i = None
    precision = 1000
    for r in range(precision):
        r_i = lower_bound + (upper_bound - lower_bound) * r / precision
        diff = abs(calculate_product_d_uniform(l_i, h_i, r_i, l_n, h_n) - (1/n)**(1/(n-1)))
        if diff < min_diff:
            min_diff = diff
            min_r_i = r_i
    return min_r_i, min_diff

# Binary search to find r_i such that G(r_i) = 1/n
# tolerance = 1e-3
mu = 0.5
sigma = 0.1
low = [0.4, 0.3, 0.79]
high = [0.7, 0.6, 0.8]
# low = [0.3, 0.8, 0.999]
# high = [0.4, 0.9, 1]
n=len(low)

# fix some low[i] and high[i] and find all r_j j!=i
i=0
l_n = low[i]
h_n = high[i]
r_values = [0]*n
for j in range(n):
    if j != i:
        l_i = low[j]
        h_i = high[j]        
        min_r_i, min_diff = find_r_i(l_i, h_i, l_n, h_n, mu, sigma, n)
        print(f"best value of r_i: {min_r_i}")
        print(f"best diff: {min_diff}")
        r_values[j] = min_r_i
print(r_values)

###### TESTING ########

def err(probabilities):
    # list of 1/n values 
    correct = np.full(len(probabilities), 1/len(probabilities))
    # l2 norm of the difference between the correct values and the observed values
    return np.linalg.norm(correct - probabilities)

num_samples = 100000
# Initialize counts
max_indexes = np.zeros(n)
baseline_max_indexes = np.zeros(n)

# Sample all at once
samples = np.array([sample_uniform(low[j], high[j], num_samples) for j in range(n)])
print(samples.shape)

# calculate how many times is 

# Fraction of time each i (n) is bigger than j + r_values[j] ~~ should be (1/n)^(1/(n-1))
proportion = np.mean(samples + np.array(r_values)[:, None] < samples[i], axis=1)
print("Proportion: i > j + r_j")
print(proportion)

# Calculate max values for baseline and optimized
baseline_max_values = np.max(samples, axis=0)
max_values = np.max(samples + np.array(r_values)[:, None], axis=0)

# Determine max indexes
baseline_max_indexes = np.argmax(samples, axis=0)
max_indexes = np.argmax(samples + np.array(r_values)[:, None], axis=0)
print(max_indexes.shape)
print(max_indexes[:10])

# Count occurrences
baseline_counts = np.bincount(baseline_max_indexes, minlength=n)
max_counts = np.bincount(max_indexes, minlength=n)
print(max_counts)

# Normalize counts to probabilities
baseline_probabilities = baseline_counts / num_samples
max_probabilities = max_counts / num_samples

print("Baseline")
print(err(baseline_probabilities))
print(baseline_probabilities)

print("Optimized")
print(err(max_probabilities))
print(max_probabilities)