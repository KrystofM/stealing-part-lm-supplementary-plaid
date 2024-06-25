from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import fsolve, root
import numpy as np
import random
from scipy.optimize import least_squares, minimize

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
    x = dist.ppf(1 - prob)
    
    return x

print(find_x_bigger_than_m(0, 0.5, 3, 0.5, 0.1))

# for every inteval in intervals find x bigger than m
def find_x_bigger_than_m_for_intervals(intervals, mean, std):
    for i in range(len(intervals)):
        l_i, h_i = intervals[i]
        x = find_x_bigger_than_m(l_i, h_i, len(intervals), mean, std)
        print(f"Interval {i}: {x}")
    

mean = 0.5  # assuming a mean, adjust as needed
std = 0.1   # assuming a standard deviation, adjust as needed
intervals = [(0.1, 0.5), (0.3, 0.6), (0.99, 1.0)]
find_x_bigger_than_m_for_intervals(intervals, mean, std)
N = len(intervals)
r_values = []
initial_guess = try_something(intervals)




# test all r values
# test_all_r_values(intervals, mean, std)


# Set the seed for reproducibility (optional)
random.seed(42)

# Solve for r_values
r_values = minimize(objective_function, initial_guess, args=(intervals, mean, std, N)).x
# r_values = [0.6666666666666666, 0.5555555555555556, 0.1111111111111111]
# r_values = [ 0.39828478,  0.07810846,  0.15058997,  0.43967887, -0.35862119, -0.50074656]
# r_values = root(system_of_equations, initial_guess, args=(intervals, mean, std, N)).x
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
for i in range(10000):
    max_value = 0
    max_index = 0
    baseline_max_value = 0
    baseline_max_index = 0
    for j in range(len(intervals)):
        sample = sample_truncated_normal(intervals[j][0], intervals[j][1], 1, mean, std)
        if sample > baseline_max_value:
            baseline_max_value = sample
            baseline_max_index = j
        if sample + r_values[j] > max_value:
            max_value = sample + r_values[j]
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
