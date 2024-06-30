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
size = 1000  # Increasing sample size for better accuracy

# Truncate X on [0, 0.5]
lower_X = 0.3999
upper_X = 0.4
X = truncated_normal(mean, sigma, lower_X, upper_X, size=size)
# Truncate Y on [0.5, 1]
lower_Y = 0.69
upper_Y = 1
Y = truncated_normal(mean, sigma, lower_Y, upper_Y, size=size)
# Truncate Z on [0.1, 0.2]
lower_Z = 0.1
upper_Z = 0.2
Z = truncated_normal(mean, sigma, lower_Z, upper_Z, size=size)
lower_W = 0.1
upper_W = 0.9
W = truncated_normal(mean, sigma, lower_W, upper_W, size=size)

# Function to calculate P(X > Y), P(Y > X), P(X > Z), P(Z > X), P(Y > Z), P(Z > Y), P(W > X), P(W > Y), P(W > Z)
def calculate_probabilities(X, Y, Z, W, shift_X=0, shift_Z=0, shift_W=0):
    X_shifted = X + shift_X
    Z_shifted = Z + shift_Z
    W_shifted = W + shift_W
    prob_X_greater = np.mean((X_shifted > Y) & (X_shifted > Z_shifted) & (X_shifted > W_shifted))
    prob_Y_greater = np.mean((Y > X_shifted) & (Y > Z_shifted) & (Y > W_shifted))
    prob_Z_greater = np.mean((Z_shifted > X_shifted) & (Z_shifted > Y) & (Z_shifted > W_shifted))
    prob_W_greater = np.mean((W_shifted > X_shifted) & (W_shifted > Y) & (W_shifted > Z_shifted))
    return prob_X_greater, prob_Y_greater, prob_Z_greater, prob_W_greater

# Initial probabilities without any shift
prob_X_greater, prob_Y_greater, prob_Z_greater, prob_W_greater = calculate_probabilities(X, Y, Z, W)
print(f'Initial P(X > Y): {prob_X_greater}, P(Y > X): {prob_Y_greater}, P(Z > X): {prob_Z_greater}, P(W > X): {prob_W_greater}')
# Calculate the means of X, Y, and Z
mean_X = np.mean(X)
mean_Y = np.mean(Y)
mean_Z = np.mean(Z)
mean_W = np.mean(W)
print(f'Mean of X: {mean_X}, Mean of Y: {mean_Y}, Mean of Z: {mean_Z}, Mean of W: {mean_W}')

# Calculate the difference of means
mean_difference_XY = mean_Y - mean_X
mean_difference_XZ = mean_Z - mean_X
print(f'Difference of means (Y - X): {mean_difference_XY}, (Z - X): {mean_difference_XZ}')

# Find the shift that balances the probabilities
shift_X = 0
shift_Z = 0
shift_W = 0
best_shift_X = 0
best_shift_Z = 0
best_shift_W = 0
smallest_difference = float('inf')
tolerance = 0.001

for shift_X in np.linspace(0, 1, 101):
    for shift_Z in np.linspace(0, 1, 101):
        for shift_W in np.linspace(0, 1, 101):
            prob_X_greater, prob_Y_greater, prob_Z_greater, prob_W_greater = calculate_probabilities(X, Y, Z, W, shift_X, shift_Z, shift_W)
            max_prob = max(prob_X_greater, prob_Y_greater, prob_Z_greater, prob_W_greater)
            min_prob = min(prob_X_greater, prob_Y_greater, prob_Z_greater, prob_W_greater)
            difference = max_prob - min_prob
            if difference < smallest_difference:
                smallest_difference = difference
                best_shift_X = shift_X
                best_shift_Z = shift_Z
                best_shift_W = shift_W
                best_prob_X_greater = prob_X_greater
                best_prob_Y_greater = prob_Y_greater
                best_prob_Z_greater = prob_Z_greater
                best_prob_W_greater = prob_W_greater
                best_prob_X_bigger_Y = np.mean((X + best_shift_X) > (Y))
                best_prob_X_bigger_Z = np.mean((X + best_shift_X) > (Z + best_shift_Z))
                if smallest_difference < tolerance:
                    break

print(f'Best P(X > Y): {best_prob_X_bigger_Y}, P(X > Z): {best_prob_X_bigger_Z}')
print(f'Best P(max(X)): {best_prob_X_greater}, P(max(Y)): {best_prob_Y_greater}, P(max(Z)): {best_prob_Z_greater}, P(max(W)): {best_prob_W_greater}')
print(f'Best shift X: {best_shift_X}, Z: {best_shift_Z}, W: {best_shift_W}, with smallest difference: {smallest_difference}')

# Plot the results with the shifted X and Z
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.hist(X + best_shift_X, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title(f'Shifted X on [0, 0.5] with shift {best_shift_X:.4f}')
plt.xlabel('X values')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(Y, bins=30, alpha=0.75, color='green', edgecolor='black')
plt.title('Truncated Y on [0.5, 1]')
plt.xlabel('Y values')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(Z + best_shift_Z, bins=30, alpha=0.75, color='red', edgecolor='black')
plt.title(f'Shifted Z on [0.1, 0.2] with shift {best_shift_Z:.4f}')
plt.xlabel('Z values')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()