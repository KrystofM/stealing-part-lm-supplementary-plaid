import scipy.integrate as integrate
from scipy.stats import truncnorm

# Define the limits of integration
mu = 0.5
sigma = 0.1
l_i = 0.3
h_i = 0.4
l_n = 0.8 
h_n = 0.9
n=3

# Define the CDF F_i and PDF f_n
def F_i(x):
    # Implement the CDF of P_i
    turncated_normal = truncnorm((l_i - mu) / sigma, (h_i - mu) / sigma, loc=mu, scale=sigma)
    return turncated_normal.cdf(x)

def f_n(x):
    # Implement the PDF of P_n
    turncated_normal = truncnorm((l_n - mu) / sigma, (h_n - mu) / sigma, loc=mu, scale=sigma)
    return turncated_normal.pdf(x)

# Define the function G(r_i)
def G(r_i):
    integrand = lambda y: F_i(y + r_i) * f_n(y)
    integral_value, _ = integrate.quad(integrand, l_i, h_i)
    return integral_value - 1/n

# Initial guess for r_i


# Binary search to find r_i such that G(r_i) = 1/n
# tolerance = 1e-3
lower_bound = max(l_i, l_n) - min(l_i, l_n)
upper_bound = max(h_i, h_n) - min(l_i, l_n)
# r_i = (lower_bound + upper_bound) / 2
# while True:
#     # Compute the value of G(r_i)
#     G_r_i = G(r_i)
#     print(f"G(r_i): {G_r_i}")
#     if abs(G_r_i) < tolerance:
#         break
#     if G_r_i < 0:
#         lower_bound = r_i
#     else:
#         upper_bound = r_i
#     r_i = (lower_bound + upper_bound) / 2
#     print(f"r_i: {r_i}")

# Try 10000 values between the lower and upper bounds, and find the best value that satisfies G(r_i) = 1/n
r_i = None
min_diff = float('inf')
for r in range(100):
    r_i = lower_bound + (upper_bound - lower_bound) * r / 100
    print(f"r_i: {r_i}")
    diff = abs(G(r_i))
    print(f"diff: {diff}")
    if diff < min_diff:
        min_diff = diff
    if diff < 1e-3:
        break


# r_i now contains the solution
print(f"Converged r_i: {r_i}")
