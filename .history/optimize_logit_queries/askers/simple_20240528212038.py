# %%
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, uniform
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import least_squares

# Vectorized version
def calculate_product_d_vectorized(l_i, h_i, r_i, intervals, mu, sigma, r_values, i):
    tnormI = truncnorm((l_i - mu) / sigma, (h_i - mu) / sigma, loc=mu, scale=sigma)    
    y_values = np.linspace(l_i, h_i, 10000)
    product = 1

    tnormI_pdf = tnormI.pdf(y_values)
    if np.any(np.isnan(tnormI_pdf)):
        print(f"tnormI_pdf contains NaN values for interval {i}")
        print(f"Parameters: l_i={l_i}, h_i={h_i}, mu={mu}, sigma={sigma}")

        return np.nan

    for j, (l_j, h_j) in enumerate(intervals):
        if j != i:
            tnormJ = truncnorm((l_j - mu) / sigma, (h_j - mu) / sigma, loc=mu, scale=sigma)
            cdf_values = tnormJ.cdf(y_values + r_i - r_values[j])
            min_max_values = np.minimum(np.maximum(cdf_values, 0), 1)
            integrand_values = min_max_values * tnormI_pdf
            if np.any(np.isnan(integrand_values)):
                print(f"integrand_values contains NaN values for interval {j}")
                return np.nan
            integral = np.trapz(integrand_values, y_values)  # Use trapezoidal rule for integration
            if np.isnan(integral):
                print(f"Integral is NaN for interval {j}")
                return np.nan
            product *= integral
    
    return product

def system_of_equations(r_values, intervals, mu, sigma):
    equations = []
    products = []
    first_product = calculate_product_d_vectorized(intervals[0][0], intervals[0][1], r_values[0], intervals, mu, sigma, r_values, 0)
    for i in range(1, len(intervals)):
        l_i, h_i = intervals[i]
        product = calculate_product_d_vectorized(l_i, h_i, r_values[i], intervals, mu, sigma, r_values, i)
        products.append(product)
    
    for i in range(1, len(products)):
        equations.append(products[i] - products[i-1])
     # Append the difference between the first product and the last product
    equations.append(first_product - products[-1])

    print(f"r_values: {r_values}")
    print(f"equations: {equations}")
    return equations

def find_x_bigger_than_m(l_i, h_i, n, mean, std):
    # Create the truncated normal distribution
    dist = truncnorm((l_i - mean) / std, (h_i - mean) / std, loc=mean, scale=std)
    
    # Calculate the probability of X > m
    prob = 1 / (n - 1)
    
    # Calculate the value of m
    x = dist.ppf(1 - prob)
    
    return x

def find_x_bigger_than_m_uniform_prior(l_i, h_i, n):
    # Create the unirom
    dist = uniform(l_i, h_i)
    
    # Calculate the probability of X > m
    prob = 1 / n
    
    # Calculate the value of m
    x = dist.isf(prob)

    return x


def every_one_over_n_uniform(low, high, constraints=None, real=None, error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    n = len(low)
    intervals = list(zip(low, high))

    n = len(intervals)
    r = [0]*n
    for i in range(n):
        l_i, h_i = intervals[i]
        x = find_x_bigger_than_m_uniform_prior(l_i, h_i, n)
        print(f"Interval {i}: {x}")
        r[i] = h_i - x
        print(f"r: {r[i]}")
        r[i] += 1 - h_i
        print(f"r: {r[i]}")
    r[base_top_token] = 0
    assert r[base_top_token] == 0
    return r


def every_one_over_n_normal(low, high, constraints=None, real=None, error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    n = len(low)
    mu = -0.621457 
    sigma = 0.049797073
    intervals = list(zip(low, high))

    # initial_guess = [0.5] * n
    # intervals = list(zip(low, high))
    # intervals[0] = (0.999, 1)
    # print('Intervals:', intervals)
    # r = least_squares(system_of_equations, initial_guess, args=(intervals, mu, sigma)).x

    # r[base_top_token] = 0
    # assert r[base_top_token] == 0

    n = len(intervals)
    r = [0]*n
    for i in range(n):
        l_i, h_i = intervals[i]
        x = find_x_bigger_than_m(l_i, h_i, n, mu, sigma)
        print(f"Interval {i}: {x}")
        r[i] = h_i - x
        print(f"r: {r[i]}")
        r[i] += 1 - h_i
        print(f"r: {r[i]}")
    r[base_top_token] = 0
    assert r[base_top_token] == 0
    return r

def normal_binary_search(low, high, **kwargs):
    bias = np.zeros(len(low))
    q = np.argmax(high - low)
    bias[q] = 1 - ((high + low) / 2)[q]
    return bias


def simultaneous_binary_search(low, high, **kwargs):
    # hyperrectangle relaxation    
    return 1 - (high + low) / 2


def start_one_over_n_merged(low, high, constraints=None, real=None,error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    NTOK = len(low)
    c = np.exp(-np.log(NTOK) / (NTOK - 1))
    # c fraction of each of the other intervals is above 1
    # [low, high] -> [low + r, high + r] s.t. (1 - (low + r)) = c * (high - low)
    r = 1 - (1 - c) * low - c * high
    assert r[base_top_token] == 0

    if error <= 1:
        mid = (low + high) / 2
        pos = len(constraints) % len(low)
        r = np.zeros(len(low))
        return 1 - (mid + r)
    
    return r


def start_one_over_n_with_prior(low, high, constraints=None, real=None, error=None, **kwargs):
    NTOK = len(low)
    c = np.exp(-np.log(NTOK) / (NTOK - 1))
    mu = 0.631457 
    sigma = 0.049797073

    r_l = np.zeros(NTOK) + mu - 3*sigma

    # problem if point under r_l
    l_c = np.maximum(r_l, low)

    # [low, high] -> [low + r, high + r] s.t. (1 - (l_c + r)) = c * (high - l_c)
    r = 1 - (1 - c) * l_c - c * high
    return r



def start_over_n_est_using_normal_dist(low, high, constraints=None, real=None, error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    NTOK = len(low)
    c = np.exp(-np.log(NTOK) / (NTOK - 1))
    # c fraction of each of the other intervals is above 1
    # [low, high] -> [low + r, high + r] s.t. (1 - (low + r)) = c * (high - low)
    r = 1 - (1 - c) * low - c * high
    assert r[base_top_token] == 0
    return r


def some_over_n(low, high, constraints=None, real=None, error=None, **kwargs):
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
        return min_r_i
    
    base_top_token, _ = constraints[0]
    mu = 0.631457 
    sigma = 0.049797073
    n=len(low)
    r = [0]*n
    low[base_top_token]=0.9999999
    high[base_top_token]=1
    # i=base_top_token
    # l_n = 0.99999
    # h_n = 1
    # Find token with the largest width
    w = high - low
    i = np.argmax(w)
    l_n = low[i]
    h_n = high[i]
    for j in range(n):
        if j != i:
            l_i = low[j]
            h_i = high[j]        
            min_r_i = find_r_i(l_i, h_i, l_n, h_n, mu, sigma, n)
            r[j] = min_r_i
  
    return r


def start_one_over_n_normal(low, high, constraints=None, real=None, error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    n = len(low)    
    sigma = 0.049797073
    mu = 0.621457 

    # r_l = np.zeros(n) + mu - 3*sigma
    # l_c = np.maximum(r_l, low)
    
    Phi_hi = norm.cdf(high, loc=mu, scale=sigma)
    Phi_li = norm.cdf(low, loc=mu, scale=sigma)
    
    # Combine the CDF values as per the formula
    combined_Phi = (1/n)**(1/(n-1)) * (Phi_hi - Phi_li) + Phi_li
    
    # Apply the inverse CDF (quantile function) with the same mean and standard deviation
    inverse_Phi = norm.ppf(combined_Phi, loc=mu, scale=sigma)
    
    # Calculate the final result
    r = 1 - inverse_Phi

    r[base_top_token] = 0.000001
    return r

def start_one_over_n_normal_V2(low, high, constraints=None, real=None, error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    n = len(low)    
    sigma = 0.049797073
    mu = 0.621457 

    # r_l = np.zeros(n) + mu - 3*sigma
    # l_c = np.maximum(r_l, low)
    
    Phi_hi = norm.cdf(high, loc=mu, scale=sigma)
    Phi_li = norm.cdf(low, loc=mu, scale=sigma)
    
    # Combine the CDF values as per the formula
    combined_Phi = (1/(n-1))**(1/(n-1)) * (Phi_hi - Phi_li) + Phi_li
    
    # Apply the inverse CDF (quantile function) with the same mean and standard deviation
    inverse_Phi = norm.ppf(combined_Phi, loc=mu, scale=sigma)
    
    # Calculate the final result
    r = 1 - inverse_Phi

    r[base_top_token] = 0
    assert r[base_top_token] == 0
    return r


def start_one_over_n_estimator(low, high, constraints=None, real=None,error=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    NTOK = len(low)
    c = np.exp(-np.log(NTOK) / (NTOK - 1))
    if kwargs.get('plot', False):
        c_values = []
        c_diff_values = []
        for n in range(2, NTOK+1):
            c_values.append(np.exp(-np.log(n) / (n - 1)))
            # (1/n)^(1/n+1)
            c_diff_values.append((1/n)**(1/(n-1)))
        plt.plot(range(2, NTOK+1), c_values, label='c values for different NTOK')
        plt.plot(range(2, NTOK+1), c_diff_values, label='(1/n)^(1/n-1)')
        plt.xlabel('NTOK')
        plt.ylabel('c value')
        plt.title('c values vs. NTOK')
        plt.legend()
        plt.show()
    # c fraction of each of the other intervals is above 1
    # [low, high] -> [low + r, high + r] s.t. (1 - (low + r)) = c * (high - low)
    r = 1 - (1 - c) * low - c * high
    r = 1 - (low - c * low) - c * high
    r = 1 - low + c * low - c * high
    r = 1 - low - c * (high - low)
    assert r[base_top_token] == 0

    # plot mid and use low and high as error bars
    if kwargs.get('plot', False):        
        mid = (low + high) / 2
        order = np.argsort(mid)
        srt_low = low[order]
        srt_high = high[order]
        srt_mid = mid[order]
        print(srt_low.shape, srt_high.shape, srt_mid.shape)
        plt.figure(figsize=(15, 8))  # Set the figure size to full screen
        plt.errorbar(range(len(srt_mid)), srt_mid, yerr=[srt_mid - srt_low, srt_high - srt_mid], fmt='o', capsize=4, color='blue', ecolor='black', capthick=2, elinewidth=2)
        plt.fill_between(range(NTOK), [1 - r[i] for i in range(NTOK)], [1 for _ in range(NTOK)], color='lightcoral', label='Bias')        
        plt.scatter(range(NTOK), srt_mid, color='blue', label='Mid', marker='o')
        real = np.sort(real)
        plt.scatter(range(len(real)), real, color='yellow', label='Real', marker='^')
        plt.plot(c, label='c')
        plt.title(f"Distribution with Area Error: {error}")
        
        plt.ylim(0, 1)   
        plt.legend()
        plt.show()

    return r


def normal_perfect(low, high, constraints=None, real=None, error=None, precision=None, order=None, **kwargs):
    # plot mid and use low and high as error bars
    # introduce random error to real
    real_order = np.argsort(real)
    real = real[real_order]
    r = np.zeros(len(low))
    pos = order[(len(constraints) - 1) % len(low)] if order is not None else len(constraints) % len(low)
    print(pos)
    r[pos] -= precision if precision is not None else 0.01
    pos_r = np.where(real_order == pos)[0][0]
    print(pos_r)
    if kwargs.get('plot', False):
        NTOK = len(low)
        mid = (low + high) / 2
        order = np.argsort(mid)
        srt_low = low[order]
        srt_high = high[order]
        srt_mid = mid[order]        
        plt.figure(figsize=(15, 8))  # Set the figure size to full screen
        plt.errorbar(range(len(srt_mid)), srt_mid, yerr=[srt_mid - srt_low, srt_high - srt_mid], fmt='o', capsize=4, color='blue', ecolor='black', capthick=2, elinewidth=2)
        plt.fill_between(range(NTOK), real, [1 for _ in range(NTOK)], color='lightcoral', label='Bias')        
        plt.scatter(range(NTOK), srt_mid, color='blue', label='Mid', marker='o')
        plt.scatter(range(len(real)), real, color='yellow', label='Real', marker='^')
        plt.title(f"Distribution with Area Error: {error}")
        plt.scatter([pos_r], [srt_mid[pos_r]], color='red', label='Last Point', marker='x', zorder=3)
        
        plt.ylim(0, 1)   
        plt.legend()
        plt.show()
    
    # add a small error to each position in rotating order    

    # find the position with the highest high - low
    # q = np.argmax(high - low)
    # r[q] -= 0.01
    # random vector
    decay_factor = 1 / (len(constraints))
    # r = np.random.uniform(-0.01 * decay_factor, 0.01 * decay_factor, len(low))
    return 1 - (real[np.argsort(real_order)]+ r)
    
def normal_distribution(low, high, constraints=None, real=None, **kwargs):
    mid = (low + high) / 2
    
    # mu = 0.721457 
    # sigma = 0.049797073
    mu = 0.6310459113121 - np.random.normal(0, 0.05) # sharp plus random small error
    sigma = 0.0457189455628 # sharp
    if kwargs.get('warm', False):
        mu = np.mean(mid)
        sigma = np.std(mid)
    # when low small this does not have to be a great approximation
    samples = np.random.normal(mu, sigma, len(low))
    samples = np.sort(samples)
    # sort mid vector and with that also low and high
    order = np.argsort(mid)
    srt_low = low[order]
    srt_high = high[order]
    srt_mid = mid[order]

     # guess mid as a sample from the normal distribution, but has to be in low and high, orderwise half the guess
    guess = samples
    # guess = guess + 0.1
    for i in range(len(guess)):
        if guess[i] < srt_low[i] or guess[i] > srt_high[i]:
            guess[i] = (srt_low[i] + srt_high[i]) / 2

    # plot mid and use low and high as error bars
    if kwargs.get('plot', False):
        plt.figure(figsize=(15, 8))  # Set the figure size to full screen
        plt.errorbar(range(len(srt_mid)), srt_mid, yerr=[srt_mid - srt_low, srt_high - srt_mid], fmt='o', capsize=4, color='blue', ecolor='black', capthick=2, elinewidth=2)
        plt.fill_between(range(len(samples)), guess, [1 for _ in range(len(samples))], color='lightcoral', label='Bias')        
        plt.scatter(range(len(samples)), samples, color='blue', label='Mid', marker='o')
        real = np.sort(real)
        plt.scatter(range(len(real)), real, color='yellow', label='Real', marker='^')
        plt.scatter(range(len(guess)), guess, color='red', label='Guess', marker='s')
        
        plt.ylim(0, 1)   
        plt.legend()
        plt.show()

    base_top_token, _ = constraints[0]
    r = (1 - guess[np.argsort(order)])
    r[base_top_token] = 0
    assert r[base_top_token] == 0
    # sort back to original order
    return r


def kde_distribution(low, high, constraints=None, real=None, **kwargs):
    mid = (low + high) / 2
    
    # To load the KDE object later
    with open('./kde_model.pkl', 'rb') as file:
        loaded_kde = pickle.load(file)
    # when low small this does not have to be a great approximation
    samples = loaded_kde.resample(size=len(low))[0]
    samples = np.sort(samples)
    # sort mid vector and with that also low and high
    order = np.argsort(mid)
    srt_low = low[order]
    srt_high = high[order]
    srt_mid = mid[order]

     # guess mid as a sample from the normal distribution, but has to be in low and high, orderwise half the guess
    guess = samples
    guess = guess + 0.1
    for i in range(len(guess)):
        if guess[i] < srt_low[i] or guess[i] > srt_high[i]:
            guess[i] = (srt_low[i] + srt_high[i]) / 2

    # plot mid and use low and high as error bars
    if kwargs.get('plot', False):
        plt.figure(figsize=(15, 8))  # Set the figure size to full screen
        plt.errorbar(range(len(srt_mid)), srt_mid, yerr=[srt_mid - srt_low, srt_high - srt_mid], fmt='o', capsize=4, color='blue', ecolor='black', capthick=2, elinewidth=2)
        plt.fill_between(range(len(samples)), guess, [1 for _ in range(len(samples))], color='lightcoral', label='Bias')        
        plt.scatter(range(len(samples)), samples, color='blue', label='Mid', marker='o')
        real = np.sort(real)
        plt.scatter(range(len(real)), real, color='yellow', label='Real', marker='^')
        plt.scatter(range(len(guess)), guess, color='red', label='Guess', marker='s')
        
        plt.ylim(0, 1)   
        plt.legend()
        plt.show()

    # sort back to original order
    return 1 - guess[np.argsort(order)]

def hyperrectangle_actual_center(
    low, high, constraints=None, distribution="uniform", **kwargs
):
    """
    hypercube relaxation, but correctly
    we want f_i(v) = P[x_i + v_i >= max_{j != i} x_j + v_j] to be 1/n, for all i
    it is true that
    f_i = \int_{low[i]}^{high[i]} \prod_{j != i} P[x_j <= x_i + v_i - v_j] dx_i

    for uniform distribution, there is an exact integral for f_i, which we approximate by sampling
    similarly for normal distribution. however we don't have mu and sigma yet, so raise notimplementederror

    we actually do logf_i = \sum_{j != i} log P[x_j <= x_i + v_i - v_j]
    """

    raise NotImplementedError


# %%
