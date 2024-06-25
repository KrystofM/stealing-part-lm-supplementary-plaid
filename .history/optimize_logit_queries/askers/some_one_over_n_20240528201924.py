import numpy as np
import scipy.integrate as integrate
from scipy.stats import truncnorm

# Define the limits of integration
# mu = 0.5
# sigma = 0.1
# l_i = 0.3
# h_i = 0.4
# l_n = 0.8 
# h_n = 0.9
# n=3

# # Define the CDF F_i and PDF f_n
# def F_i(x):
#     # Implement the CDF of P_i
#     turncated_normal = truncnorm((l_i - mu) / sigma, (h_i - mu) / sigma, loc=mu, scale=sigma)
#     return turncated_normal.cdf(x)

# def f_n(x):
#     # Implement the PDF of P_n
#     turncated_normal = truncnorm((l_n - mu) / sigma, (h_n - mu) / sigma, loc=mu, scale=sigma)
#     return turncated_normal.pdf(x)

# # Define the function G(r_i)
# def G(r_i):
#     integrand = lambda y: F_i(y - r_i) * f_n(y)
#     integral_value, _ = integrate.quad(integrand, l_i, h_i)
#     return integral_value - 1/n

# Vectorized version
# P(P_i < P_n - r_i)
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
    print(f"lower_bound: {lower_bound}")
    print(f"upper_bound: {upper_bound}")
    r_i = None
    min_diff = float('inf')
    min_r_i = None
    precision = 100
    for r in range(precision):
        r_i = lower_bound + (upper_bound - lower_bound) * r / precision
        diff = abs(calculate_product_d_vectorized(l_i, h_i, r_i, l_n, h_n, mu, sigma) - 1/n)
        if diff < min_diff:
            min_diff = diff
            min_r_i = r_i
    return min_r_i, min_diff

# Binary search to find r_i such that G(r_i) = 1/n
# tolerance = 1e-3
mu = 0.5
sigma = 0.1
low = [0.3, 0.8, 0.999]
high = [0.4, 0.9, 1]
n=3
min_r_i, min_diff = find_r_i(l_i, h_i, l_n, h_n, mu, sigma, n)
print(f"best value of r_i: {min_r_i}")
print(f"best diff: {min_diff}")
