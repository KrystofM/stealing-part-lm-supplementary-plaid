import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt

# Function to generate truncated normal samples
def truncated_normal(mean, sigma, lower, upper, size=1):
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    truncated_distribution = stats.truncnorm(a, b, loc=mean, scale=sigma)
    samples = truncated_distribution.rvs(size=size)
    return samples

# Hyper Parameters
mean = 0.5
sigma = 0.1
sample_size = 100000  # Increasing sample size for better accuracy
variable_count = 15
initial_radius = (1 / variable_count) # 2 
decay = 0.95
iterations = 200
interation_sample_size = 25

# lower array of lower bounds random numbers
# set seed
# np.random.seed(45)
# hundred tuple random numbers between 0 and 1
lower = np.random.uniform(0, 0.99, size=variable_count)
# for each lower value make a new upper value that is random between lower and 1
upper = np.random.uniform(lower, 1, size=variable_count)
print(f'Lower: {lower}')
print(f'Upper: {upper}')

# lower = np.array([0.37079472 , 0.15445845,0.94120716, 0.16443458, 0.724674])
# upper = np.array([0.40734123, 0.1718635, 0.99213212, 0.97455681, 0.89017659])
# variable_count = len(lower)

# lower =  np.random.uniform(0, 0.5, size=variable_count) # np.array([0.69,0.3999, 0.1, 0.1]) #
# lower[0] = 0.9
# # upper array of upper bounds random numbers
# upper =  np.random.uniform(0.5, 1, size=variable_count)#  np.array([1, 0.4, 0.2, 0.9]) # np.random.uniform(0.5, 1, size=size)
#upper[0] = 1
# truncated normal random variables array
X = np.array([truncated_normal(mean, sigma, l, u, size=sample_size) for l, u in zip(lower, upper)])
print(X.shape)

# Function to calculate probability of each random varible being the maximum in X after shifting
# X is a list of random variables
def calculate_probabilities(X, shift):
    X_shifted = X + shift.reshape(-1, 1)

    max_p = []
    for i in range(len(X)):
        condition = []
        for j in range(len(X)):
            if i != j:
                condition.append(X_shifted[i] > X_shifted[j])
        max_p.append(np.mean(np.all(condition, axis=0)))
    return max_p

print(f'Initial probabilities: {calculate_probabilities(X, np.zeros(variable_count))}')
means = np.mean(X, axis=1)
print(f'Means: {means}')
std = np.std(X, axis=1)

def loss_function(probabilities):    
    if len(probabilities) != variable_count:
        raise ValueError(f"Length of probabilities ({len(probabilities)}) does not match variable count ({variable_count})")
    
    if any(prob == 0 for prob in probabilities):
        return float('inf')    
    
    diff = abs(probabilities - np.ones(variable_count) / (variable_count))
    diff = np.exp(diff * 100)
       
    return np.linalg.norm(diff, ord=1)

def start_over_n_shift(low, high, mu, sigma):
    Phi_hi = norm.cdf(high, loc=mu, scale=sigma)
    Phi_li = norm.cdf(low, loc=mu, scale=sigma)
    
    # Combine the CDF values as per the formula
    combined_Phi = (1/variable_count)**(1/(variable_count-1)) * (Phi_hi - Phi_li) + Phi_li
    
    # Apply the inverse CDF (quantile function) with the same mean and standard deviation
    inverse_Phi = norm.ppf(combined_Phi, loc=mu, scale=sigma)
    
    # Calculate the final result
    r = 1 - inverse_Phi

    return r

start_over_n_shift_initial = start_over_n_shift(lower, upper, mean, sigma)
print(f'Start over n shift: {start_over_n_shift_initial}')
print(f'Start over n shift probabilities: {calculate_probabilities(X, start_over_n_shift_initial)}')
print(f'Start over n shift loss: {loss_function(calculate_probabilities(X, start_over_n_shift_initial))}')

# # Fix the first variable and take the difference of means to find the shift
# fixed_mean = means[0]
# fixed_sigma = std[0]
# est_shift = fixed_mean - means
# est_shift[0] = 0  # No shift for the first variable
# print(f'Estimated shift: {est_shift}')
# print(f'Estimated shift probabilities: {calculate_probabilities(X, est_shift)}')
# print(f'Estimated shift loss: {loss_function(calculate_probabilities(X, est_shift))}')

# create a hypersphere with radius R around the estimated shift and sample from it
def optimize_random_search(X, initial_guess):
    R = initial_radius
    # for each point, calculate the probability that the point is the maximum in X
    best_distance = [float('inf')]
    best_point = [initial_guess]
    best_probabilities = [calculate_probabilities(X, initial_guess)]
    no_improvement_count = 0
    speedup = 0 
    for iteration in range(iterations):
        if no_improvement_count >= 5:
            speedup = speedup + 1  # Faster decay
        else:
            speedup = speedup
        # points on the hypersphere
        radius = R * (decay ** (iteration + speedup))
        points = np.random.uniform(0, radius, (interation_sample_size, len(lower)))
        # add the estimated shift to each point (center the hypersphere)
        points += best_point[-1].reshape(1, -1)   
        best_distance_round = loss_function(calculate_probabilities(X, best_point[-1]))
        best_point_round = best_point[-1]
        best_probabilities_round = best_probabilities[-1]

        print(points.shape)       
        for point in points:
            probabilities = calculate_probabilities(X, point)
            # distance from 1/n probabilities to the point probabilities
            distance = loss_function(probabilities)
            if distance < best_distance_round:
                best_point_round = point
                best_probabilities_round = probabilities
                best_distance_round = distance

        if best_distance_round == best_distance[-1]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        best_point.append(best_point_round)
        best_probabilities.append(best_probabilities_round)
        best_distance.append(best_distance_round)
        print(f'Iteration: {iteration}, Loss: {best_distance_round}, Radius: {radius}')


    print(f'Best point: {best_point[-1]}, probabilities: {best_probabilities[-1]}')
    return best_point[-1]
# optimal_shift = optimize_random_search(X, start_over_n_shift_initial)
# print(f'Optimal shift: {optimal_shift}')
# print(f'Optimal shift probabilities: {calculate_probabilities(X, optimal_shift)}')
# print(f'Optimal loss: {loss_function(calculate_probabilities(X, optimal_shift))}')

def everything_one_over_n_normal(low, high, constraints=None, real=None, **kwargs):
    mean = 0.6884241
    sigma = 0.06653234
    # ger rid of the first element of low and high
    low[0] = 0.999999999999999
    high[0] = 1
    start_over_n_shift_initial = start_over_n_shift(low, high, mean, sigma)
    print(low)
    print(high)
    print(start_over_n_shift_initial)
    X = np.array([truncated_normal(mean, sigma, l, u, size=sample_size) for l, u in zip(low, high)])
    optimal_shift = optimize_random_search(X, start_over_n_shift_initial)

    return optimal_shift


def optimize_shift_gradient(X, initial_shift, iterations=1000, step_size=0.1):
    current_shift = np.array(initial_shift)
    best_shift = current_shift.copy()
    best_loss = loss_function(calculate_probabilities(X, best_shift))
    losses = [best_loss]  # Store initial loss
    previous_loss = best_loss  # Initialize previous loss
    no_improvement_count = 0  # Initialize no improvement counter
    current_step_size = step_size

    for iteration in range(iterations):
        print(f'Iteration: {iteration}, Loss: {best_loss}, No improvement count: {no_improvement_count}')
        # Dynamic step size based on loss improvement
        
        loss_improvement = previous_loss - best_loss
        previous_loss = best_loss  # Update previous loss
        print(f'Loss improvement: {loss_improvement}')
        if loss_improvement <= 0:
            current_step_size = current_step_size * 0.8  # Reduce step size if no improvement
            no_improvement_count += 1  # Increment no improvement counter
        else:
            current_step_size = step_size  # Maintain step size if there is improvement
            no_improvement_count = 0  # Reset no improvement counter

        # Break the loop if no improvement for specified consecutive rounds
        if no_improvement_count >= 30:
            print(f"Breaking after {iteration+1} iterations due to no improvement.")
            break

        for i in range(1, len(current_shift)):
            # Explore in both directions for each dimension
            temp_shift = current_shift.copy()
            temp_shift[i] += current_step_size
            loss_plus = loss_function(calculate_probabilities(X, temp_shift))
            
            temp_shift[i] -= 2 * current_step_size
            loss_minus = loss_function(calculate_probabilities(X, temp_shift))
            
            # Choose the direction with the smallest loss
            if loss_plus < loss_minus and loss_plus < best_loss:
                best_loss = loss_plus
                best_shift[i] += current_step_size
            elif loss_minus < best_loss:
                best_loss = loss_minus
                best_shift[i] -= current_step_size

        # Update current shift to the best found in this iteration
        current_shift = best_shift.copy()
        losses.append(best_loss)  # Append the best loss of this iteration
        

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve during Gradient Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_shift


# optimal_shift = optimize_shift_gradient(X, start_over_n_shift_initial)
# print(f'Optimal shift: {optimal_shift}')
# print(f'Optimal shift probabilities: {calculate_probabilities(X, optimal_shift)}')
# print(f'Optimal loss: {loss_function(calculate_probabilities(X, optimal_shift))}')