import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Function to generate truncated normal samples
def truncated_normal(mean, sigma, lower, upper, size=1):
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    truncated_distribution = stats.truncnorm(a, b, loc=mean, scale=sigma)
    samples = truncated_distribution.rvs(size=size)
    return samples

# Parameters
mean = 0.5
sigma = 0.8
size = 10000  # Increasing sample size for better accuracy

# Truncate X on [0, 0.5]
lower_X = 0
upper_X = 0.5
X = truncated_normal(mean, sigma, lower_X, upper_X, size=size)

# Truncate Y on [0.5, 1]
lower_Y = 0.5
upper_Y = 1
Y = truncated_normal(mean, sigma, lower_Y, upper_Y, size=size)

# Function to calculate P(X > Y) and P(Y > X)
def calculate_probabilities(X, Y, shift=0):
    X_shifted = X + shift
    prob_X_greater = np.mean(X_shifted > Y)
    prob_Y_greater = np.mean(Y > X_shifted)
    return prob_X_greater, prob_Y_greater

# Initial probabilities without any shift
prob_X_greater, prob_Y_greater = calculate_probabilities(X, Y)
print(f'Initial P(X > Y): {prob_X_greater}, P(Y > X): {prob_Y_greater}')

# Find the shift that makes P(X > Y) equal to P(Y > X)
shift = 0
tolerance = 0.001
max_iterations = 1000
for i in range(max_iterations):
    prob_X_greater, prob_Y_greater = calculate_probabilities(X, Y, shift)
    if abs(prob_X_greater - prob_Y_greater) < tolerance:
        break
    shift += 0.0001 if prob_X_greater < prob_Y_greater else -0.0001

print(f'Shift: {shift}')
prob_X_greater, prob_Y_greater = calculate_probabilities(X, Y, shift)
print(f'Final P(X > Y): {prob_X_greater}, P(Y > X): {prob_Y_greater}')

# Plot the results with the shifted X
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(X + shift, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title(f'Shifted X on [0, 0.5] with shift {shift:.4f}')
plt.xlabel('X values')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(Y, bins=30, alpha=0.75, color='green', edgecolor='black')
plt.title('Truncated Y on [0.5, 1]')
plt.xlabel('Y values')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
