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
sigma = 0.1
size = 10000  # Increasing sample size for better accuracy

# Truncate X on [0, 0.5]
lower_X = 0.1
upper_X = 0.9
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
shift_values = np.linspace(0, 1, 10001)  # Generate 1001 shift values from 0 to 1
best_shift = 0
smallest_difference = float('inf')  # Initialize with a large number
best_prob_X_greater = 0  # Initialize the best P(X > Y)
best_prob_Y_greater = 0  # Initialize the best P(Y > X)

for shift in shift_values:
    prob_X_greater, prob_Y_greater = calculate_probabilities(X, Y, shift)
    difference = abs(prob_X_greater - prob_Y_greater)
    if difference < smallest_difference:
        smallest_difference = difference
        best_shift = shift
        best_prob_X_greater = prob_X_greater  # Update the best P(X > Y)
        best_prob_Y_greater = prob_Y_greater  # Update the best P(Y > X)
        if smallest_difference < tolerance:
            break

# Calculate the means of X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)
print(f'Mean of X: {mean_X}, Mean of Y: {mean_Y}')

# Calculate the difference of means
mean_difference = mean_Y - mean_X
print(f'Difference of means (Y - X): {mean_difference}')

print(f'Best P(X > Y): {best_prob_X_greater}, P(Y > X): {best_prob_Y_greater}')
print(f'Best shift: {best_shift}, with smallest difference: {smallest_difference}')
# Plot the results with the shifted X
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(X + best_shift, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title(f'Shifted X on [0, 0.5] with shift {best_shift:.4f}')
plt.xlabel('X values')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(Y, bins=30, alpha=0.75, color='green', edgecolor='black')
plt.title('Truncated Y on [0.5, 1]')
plt.xlabel('Y values')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
