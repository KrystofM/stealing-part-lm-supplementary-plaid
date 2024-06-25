from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import uniform
from scipy.integrate import quad
from scipy.optimize import fsolve, root
import numpy as np
import random
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

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
    y_values = np.linspace(l_i, h_i, 100000)  # Adjust the number of points for accuracy/speed trade-off
    product = 1

    tnormI_pdf = tnormI.pdf(y_values)

    for j, (l_j, h_j) in enumerate(intervals):
        if j != i:
            tnormJ = truncnorm((l_j - mu) / sigma, (h_j - mu) / sigma, loc=mu, scale=sigma)
            # print(f"l_i: {l_i}, h_i: {h_i}, r_i: {r_i}, l_j: {l_j}, h_j: {h_j}, r_j: {r_values[j]}")            
            cdf_values = tnormJ.cdf(y_values + r_i - r_values[j])
            min_max_values = np.minimum(np.maximum(cdf_values, 0), 1)
            integrand_values = min_max_values * tnormI_pdf
            # print(tnormI.pdf(y_values))
            # print(tnormJ.cdf(y_values + r_i - r_values[j]))
            # print pdf of tnormI at half point
            # print(tnormI.pdf((l_i + h_i) / 2))
            integral = np.trapz(integrand_values, y_values)  # Use trapezoidal rule for integration
            product *= integral
    
    return product

def sample_truncated_normal(l_i, h_i, n, mean, std):
    # Create the truncated normal distribution
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    
    # Sample n many samples from the truncated normal distribution
    samples = dist.rvs(size=n)
    
    return samples

def sample_truncated_normal_adjusted(l_i, h_i, n, mean, std):
    width = h_i - l_i
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    samples = dist.rvs(size=n)
    return samples / width  # Adjust by the width of the interval

def calculate_product_d_simulated(l_i, h_i, r_i, intervals, mu, sigma, r_values, i):
    # Sample from all truncated normal distributions and calculate how many time is i max
    n = 50
    max_indexes = {i: 0 for i in range(len(intervals))}
    for _ in range(n):
        max_value = 0
        max_index = 0
        for j in range(len(intervals)):
            sample = sample_truncated_normal(intervals[j][0], intervals[j][1], 1, mu, sigma) + r_values[j]
            if sample > max_value:
                max_value = sample
                max_index = j
        max_indexes[max_index] = max_indexes.get(max_index, 0) + 1
    return max_indexes[i] / n

def test_calculute_products():
    # simple_intervals = [(0, 0.5), (0.5, 1)]
    # simple_mu = 0.5
    # simple_sigma = 0.1
    # simple_r_values = [0, 0]
    # simple_i = 0
    # simple_product_d_vectorized = calculate_product_d_vectorized(simple_intervals[0][0], simple_intervals[0][1], simple_r_values[0], simple_intervals, simple_mu, simple_sigma, simple_r_values, simple_i)
    # print('Simple product d vectorized:', simple_product_d_vectorized)
    # simple_product_d_vectorized = calculate_product_d_vectorized(simple_intervals[1][0], simple_intervals[1][1], simple_r_values[1], simple_intervals, simple_mu, simple_sigma, simple_r_values, 1)
    # print('Simple product d vectorized:', simple_product_d_vectorized)
    simple_intervals = [(0, 0.5), (0.5, 1)]
    simple_mu = 0.5
    simple_sigma = 0.1
    simple_r_values = [0.5, 0]
    simple_i = 0
    simple_product_d_vectorized = calculate_product_d_vectorized(simple_intervals[0][0], simple_intervals[0][1], simple_r_values[0], simple_intervals, simple_mu, simple_sigma, simple_r_values, simple_i)
    print('Simple product d vectorized:', simple_product_d_vectorized)
    # simulate
    simple_product_d_simulated = calculate_product_d_simulated(simple_intervals[0][0], simple_intervals[0][1], simple_r_values[0], simple_intervals, simple_mu, simple_sigma, simple_r_values, simple_i)
    print('Simple product d simulated:', simple_product_d_simulated)
    simple_intervals = [(0.5, 1), (0.5, 1)]
    simple_mu = 0.5
    simple_sigma = 1
    simple_r_values = [0, 0]
    simple_i = 0
    simple_product_d_vectorized = calculate_product_d_vectorized(simple_intervals[0][0], simple_intervals[0][1], simple_r_values[0], simple_intervals, simple_mu, simple_sigma, simple_r_values, simple_i)
    print('Simple product d vectorized:', simple_product_d_vectorized)
    # simulate
    simple_product_d_simulated = calculate_product_d_simulated(simple_intervals[0][0], simple_intervals[0][1], simple_r_values[0], simple_intervals, simple_mu, simple_sigma, simple_r_values, simple_i)
    print('Simple product d simulated:', simple_product_d_simulated)
    # simple_intervals = [(0, 0.5), (0.5, 1)]
    # simple_mu = 0.25
    # simple_sigma = 0.25
    # simple_r_values = [0.5, 0]
    # simple_i = 0
    # simple_product_d_vectorized = calculate_product_d_vectorized(simple_intervals[0][0], simple_intervals[0][1], simple_r_values[0], simple_intervals, simple_mu, simple_sigma, simple_r_values, simple_i)
    # print('Simple product d vectorized:', simple_product_d_vectorized)

def system_of_equations(r_values, intervals, mu, sigma, n):
    equations = []
    products = []
    first_product = calculate_product_d_vectorized(intervals[0][0], intervals[0][1], r_values[0], intervals, mu, sigma, r_values, 0)
    for i in range(len(intervals)):
        l_i, h_i = intervals[i]
        product = calculate_product_d_simulated(l_i, h_i, r_values[i], intervals, mu, sigma, r_values, i)
        products.append(product)
    
    for i in range(len(products)):
        if i != 0:
            equations.append(products[0] - products[i])

    # equations.append(products[-1] - 1/n)

    print(f"r_values: {r_values}")
    print(f"equations: {equations}")
    return equations

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

def objective_function(r_values, intervals, mu, sigma, n):
    equations = system_of_equations(r_values, intervals, mu, sigma, n)
    return np.sum(np.square(equations))

def test_all_r_values(intervals, mu, sigma):
    v = np.linspace(0, 1, 10)
    min_obj_value = float('inf')
    best_r_values = None
    # all permutations of 3 time v 
    for i in v:
        for j in v:
            for k in v:
                r_values = [i, j, k]
                obj_value = objective_function(r_values, intervals, mu, sigma, 3)
                if obj_value < min_obj_value:
                    min_obj_value = obj_value
                    best_r_values = r_values
    print(f"Best r_values: {best_r_values}")
    print(f"Lowest objective_function: {min_obj_value}")

# find at what value of [li, hi] the interval has a chance of being in [m, hi] where m> li with a probability of 1/n
# find P(X > m) = 1/n <=> P(m < X) = 1/n for a truncated normal distribution
def find_x_bigger_than_m(l_i, h_i, n, mean, std):
    # Create the truncated normal distribution
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    
    # Calculate the probability of X > m
    prob = 1 / n
    
    # Calculate the value of m
    x = dist.isf(prob)

    return x

def find_x_bigger_than_m_uniform_prior(l_i, h_i, n, mean, std):
    # Create the unirom
    dist = uniform(l_i, 1)
    
    # Calculate the probability of X > m
    prob = 1 / n
    
    # Calculate the value of m
    x = dist.isf(prob)

    return x

def sample_uniform(l_i, h_i, n, mean, std):
    dist = uniform(l_i, 1)
    samples = dist.rvs(size=n)
    return samples

# for every inteval in intervals find x bigger than m
def find_x_bigger_than_m_for_intervals(intervals, mean, std):
    n = len(intervals)
    r = [0]*n
    for i in range(n):
        l_i, h_i = intervals[i]
        x = find_x_bigger_than_m(l_i, h_i, n, mean, std)
        print(f"Interval {i}: {x}")
        r[i] = h_i - x
        print(f"r: {r[i]}")
        r[i] += 1 - h_i
        print(f"r: {r[i]}")
    return r
    

def test_find_x_bigger_than_m():
    l_i = 0.99
    h_i = 1
    n = 3
    mean = 0.5
    std = 0.1
    x = find_x_bigger_than_m(l_i, h_i, n, mean, std)
    
    # sample from the truncated normal distribution
    samples = sample_truncated_normal(l_i, h_i, 10000, mean, std)
    # look for the percentage of samples that are bigger than x
    count = 0
    for sample in samples:
        if sample > x:
            count += 1
    print(f"Percentage of samples bigger than x: {count/10000}")

# test_find_x_bigger_than_m()

mean = 1.5  # assuming a mean, adjust as needed
std = 0.1   # assuming a standard deviation, adjust as needed
intervals = [(0.4, 0.5), (0.5, 0.6), (0.7, 0.8), (0.3, 0.4)]
N = len(intervals)
r_values = []
initial_guess = try_something(intervals)
test_find_x_bigger_than_m()
# test all r values
# test_all_r_values(intervals, mean, std)


# Set the seed for reproducibility (optional)
random.seed(42)

# Solve for r_values
# r_values = minimize(objective_function, initial_guess, args=(intervals, mean, std, N)).x
# r_values = [0.6666666666666666, 0.5555555555555556, 0.1111111111111111]
# r_values = [ 0.39828478,  0.07810846,  0.15058997,  0.43967887, -0.35862119, -0.50074656]
# r_values = root(system_of_equations, initial_guess, args=(intervals, mean, std, N)).x
# r_values = try_something(intervals)
r_values = find_x_bigger_than_m_for_intervals(intervals, mean, std)
print("r_values:", r_values)


# add r_values to l_i and h_i in intervals
for i in range(len(intervals)):
    l_i, h_i = intervals[i]
    r_i = r_values[i]
    intervals[i] = (l_i, h_i)

print(intervals)

def err(probabilities):
    # list of 1/n values 
    correct = np.full(len(probabilities), 1/len(probabilities))
    # l2 norm of the difference between the correct values and the observed values
    return np.linalg.norm(correct - probabilities)

num_samples = 50000
# Test python random function
samples = [random.randint(0, 10) for _ in range(num_samples)]

# Calculate the frequency of each number
frequency = {i: samples.count(i) for i in range(11)}

# Convert frequency to probability
probability = {k: v / num_samples for k, v in frequency.items()}

# Convert dictionary to numpy array
probability_array = np.array(list(probability.values()))

print("Random function")
print(err(probability_array))

num_intervals = len(intervals)

# Initialize counts
max_indexes = np.zeros(num_intervals)
baseline_max_indexes = np.zeros(num_intervals)

# Sample all at once
samples = np.array([sample_truncated_normal(interval[0], interval[1], num_samples, mean, std) for interval in intervals])

# Calculate max values for baseline and optimized
baseline_max_values = np.max(samples, axis=0)
max_values = np.max(samples + np.array(r_values)[:, None], axis=0)

# Determine max indexes
baseline_max_indexes = np.argmax(samples, axis=0)
max_indexes = np.argmax(samples + np.array(r_values)[:, None], axis=0)

# Count occurrences
baseline_counts = np.bincount(baseline_max_indexes, minlength=num_intervals)
max_counts = np.bincount(max_indexes, minlength=num_intervals)

# Normalize counts to probabilities
baseline_probabilities = baseline_counts / num_samples
max_probabilities = max_counts / num_samples

print("Baseline")
print(err(baseline_probabilities))
print(baseline_probabilities)

print("Optimized")
print(err(max_probabilities))
print(max_probabilities)