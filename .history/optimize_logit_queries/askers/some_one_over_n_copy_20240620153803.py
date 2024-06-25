import numpy as np
import scipy.integrate as integrate
from scipy.stats import truncnorm

def sample_truncated_normal(l_i, h_i, n, mean, std):
    # Create the truncated normal distribution
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    
    # Sample n many samples from the truncated normal distribution
    samples = dist.rvs(size=n)
    
    return samples

# Vectorized version
# P(P_i + r_i < P_n) = (1/n)^(1/(n-1)
def calculate_product_d_vectorized(l_i, h_i, r_i, l_n, h_n, mu, sigma):
    tnormI = truncnorm((l_i - mu) / sigma, (h_i - mu) / sigma, loc=mu, scale=sigma)    
    tnormN = truncnorm((l_n - mu) / sigma, (h_n - mu) / sigma, loc=mu, scale=sigma)     

    y_values = np.linspace(l_n, h_n, 1000)  # Adjust the number of points for accuracy/speed trade-off
    tnormN_pdf = tnormN.pdf(y_values)     
    cdf_values = tnormI.cdf(y_values - r_i)
    min_max_values = np.minimum(np.maximum(cdf_values, 0), 1)
    integrand_values = min_max_values * tnormN_pdf
    integral = np.trapz(integrand_values, y_values)  # Use trapezoidal rule for integration
    
    return integral
 
def find_r_i(l_i, h_i, l_n, h_n, mu, sigma, n):
    lower_bound = l_n - h_i
    upper_bound = h_n - l_i
    r_i = None
    min_diff = float('inf')
    min_r_i = None
    precision = 100
    for r in range(precision):
        r_i = lower_bound + (upper_bound - lower_bound) * r / precision
        diff = abs(calculate_product_d_vectorized(l_i, h_i, r_i, l_n, h_n, mu, sigma) - (1/n)**(1/(n-1)))
        if diff < min_diff:
            min_diff = diff
            min_r_i = r_i
    return min_r_i, min_diff

# Binary search to find r_i such that G(r_i) = 1/n
# tolerance = 1e-3
mu = 0.5
sigma = 0.1
low = [0.4, 0.3, 0.5]
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
samples = np.array([sample_truncated_normal(low[j], high[j], num_samples, mu, sigma) for j in range(n)])
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



# plot the pdf of the truncated normal distribution for each interval in separate plots
import matplotlib.pyplot as plt
for i in range(n):
    plt.figure(figsize=(10, 6))
    x = np.linspace(low[i], high[i], 1000)
    dist = truncnorm((low[i] - mu) / sigma, (high[i] - mu) / sigma, loc=mu, scale=sigma)
    plt.plot(x, dist.pdf(x), label=f'Interval {i+1}')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title(f'PDF of Truncated Normal Distribution for Interval {i+1}')
    plt.legend()
    plt.show()

# plot the pdfs of the shifted truncated normal distributions over each other
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, n))
for i in range(n):
    x = np.linspace(low[i], high[i], 1000)
    shifted_dist = truncnorm((low[i] - mu - r_values[i]) / sigma, (high[i] - mu - r_values[i]) / sigma, loc=mu + r_values[i], scale=sigma)
    plt.plot(x, shifted_dist.pdf(x), color=colors[i], label=f'Shifted Interval {i+1}')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('PDFs of Shifted Truncated Normal Distributions')
plt.legend()
plt.show()
