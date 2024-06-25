from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import fsolve, root
import numpy as np
import random
from scipy.optimize import least_squares

def calculate_product(l_i, h_i, r_i, intervals, mu, sigma, r_values, i):
    def integrand(y, l_j, h_j, r_j):
        return norm.pdf(y, loc=mu, scale=sigma) * (norm.cdf(y + r_i - r_j, loc=mu, scale=sigma) - norm.cdf(l_j, loc=mu, scale=sigma))
    
    product = 1
    for j, (l_j, h_j) in enumerate(intervals):
        if j != i:
            integral, _ = quad(integrand, l_i, h_i, args=(l_j, h_j, r_values[j]))
            product *= (1 / ((norm.cdf(h_j, loc=mu, scale=sigma) - norm.cdf(l_j, loc=mu, scale=sigma)) * (norm.cdf(h_i, loc=mu, scale=sigma) - norm.cdf(l_i, loc=mu, scale=sigma)))) * integral
    
    return product

def calculate_product_d(l_i, h_i, r_i, intervals, mu, sigma, r_values, i):
    def integrand(y, l_j, h_j, r_j):
        tnormI = truncnorm((l_i - mu) / sigma, (h_i - mu) / sigma, loc=mu, scale=sigma)
        tnormJ = truncnorm((l_j - mu) / sigma, (h_j - mu) / sigma, loc=mu, scale=sigma)
        return tnormJ.cdf(y + r_i - r_j) * tnormI.pdf(y)
    
    product = 1
    for j, (l_j, h_j) in enumerate(intervals):
        if j != i:
            integral, _ = quad(integrand, l_i, h_i, args=(l_j, h_j, r_values[j]))
            product *= integral
    
    return product

# Vectorized version
def calculate_product_d_vectorized(l_i, h_i, r_i, intervals, mu, sigma, r_values, i):
    tnormI = truncnorm((l_i - mu) / sigma, (h_i - mu) / sigma, loc=mu, scale=sigma)
    y_values = np.linspace(l_i, h_i, 10000)  # Adjust the number of points for accuracy/speed trade-off
    product = 1

    for j, (l_j, h_j) in enumerate(intervals):
        if j != i:
            tnormJ = truncnorm((l_j - mu) / sigma, (h_j - mu) / sigma, loc=mu, scale=sigma)
            integrand_values = tnormJ.cdf(y_values + r_i - r_values[j]) * tnormI.pdf(y_values)
            integral = np.trapz(integrand_values, y_values)  # Use trapezoidal rule for integration
            product *= integral
    
    return product

def system_of_equations(r_values, intervals, mu, sigma, n):
    equations = []
    products = []
    first_product = calculate_product(intervals[0][0], intervals[0][1], r_values[0], intervals, mu, sigma, r_values, 0)
    for i in range(1, len(intervals)):
        l_i, h_i = intervals[i]
        product = calculate_product_d_vectorized(l_i, h_i, r_values[i], intervals, mu, sigma, r_values, i)
        products.append(product)
    
    for i in range(1, len(products)):
        equations.append(products[i] - products[i-1])
     # Append the difference between the first product and the last product
    equations.append(first_product - products[-1])
    equations.append(first_product - products[-2])

    print(f"r_values: {r_values}")
    return equations

def sample_truncated_normal(l_i, h_i, n):
    # Create the truncated normal distribution
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    
    # Sample n many samples from the truncated normal distribution
    samples = dist.rvs(size=n)
    
    return samples

def mean_truncated_normal(l_i, h_i):
    # Create the truncated normal distribution
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    
    # Calculate the mean of the truncated normal distribution
    tmean = dist.mean()
    
    return tmean

def try_something(intervals):
    means = [mean_truncated_normal(l_i, h_i) for l_i, h_i in intervals]
    r_valuess = [0] * len(intervals)
    max_mean = max(means)
    for i in range(len(intervals)):
        l_i, h_i = intervals[i]
        print(f"l_i: {l_i}, h_i: {h_i}, mean: {means[i]}")
        # the wider the more likely you are to be the max -> add more to the r_value when width is smaller
        r_valuess[i] = mean - h_i

    return r_valuess



mean = 0.5  # assuming a mean, adjust as needed
std = 0.1   # assuming a standard deviation, adjust as needed
intervals = [(0.1, 0.5), (0.3, 0.6), (0.2, 0.7), (0.0, 0.4), (0.2, 0.8)]
N = len(intervals)
r_values = []
initial_guess = try_something(intervals)
print("initial_guess:", initial_guess)

# Set the seed for reproducibility (optional)
random.seed(42)

# Solve for r_values
r_values = root(system_of_equations, initial_guess, args=(intervals, mean, std, N), method='krylov').x
# r_values = try_something(intervals)
print("r_values:", r_values)


# add r_values to l_i and h_i in intervals
for i in range(len(intervals)):
    l_i, h_i = intervals[i]
    r_i = r_values[i]
    intervals[i] = (l_i, h_i)

print(intervals)

def err(max_indexes):
    # list of 1/n values 
    correct = [1/len(max_indexes) for _ in range(len(max_indexes))]
    # sum values in max_indexes
    total = sum(max_indexes.values())
    # devide each value in max_indexes by total
    observed = [value/total for value in max_indexes.values()]
    # l2 norm of the difference between the correct values and the observed values
    return np.linalg.norm(np.array(correct) - np.array(observed))


# test python random function
# Sample randomly between 0 and 10
samples = [random.randint(0, 10) for _ in range(10000)]

# Calculate the frequency of each number
frequency = {i: samples.count(i) for i in range(11)}

# Convert frequency to probability
probability = {k: v / 10000 for k, v in frequency.items()}

print(probability)
print(err(probability))

# sample from the intervals and check to which interval sample belongs the largest value
max_indexes = {i: 0 for i in range(len(intervals))}
baseline_max_indexes = {i: 0 for i in range(len(intervals))}
for i in range(2000):
    max_value = 0
    max_index = 0
    baseline_max_value = 0
    baseline_max_index = 0
    for j in range(len(intervals)):
        sample = sample_truncated_normal(intervals[j][0], intervals[j][1], 1)
        if sample > baseline_max_value:
            baseline_max_value = sample
            baseline_max_index = j
        if sample + r_values[j] > max_value:
            max_value = sample
            max_index = j        
    max_indexes[max_index] = max_indexes.get(max_index, 0) + 1
    baseline_max_indexes[baseline_max_index] = baseline_max_indexes.get(baseline_max_index, 0) + 1
    # print(err(max_indexes))

print("Baseline")
print(err(baseline_max_indexes))
total = sum(baseline_max_indexes.values())
for key in baseline_max_indexes:
    baseline_max_indexes[key] = baseline_max_indexes[key]/total
print(baseline_max_indexes)
# print the max_indexes as a percentage
print("Optimized")
print(err(max_indexes))
total = sum(max_indexes.values())
for key in max_indexes:
    max_indexes[key] = max_indexes[key]/total
print(max_indexes)
