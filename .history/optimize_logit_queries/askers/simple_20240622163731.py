# %%
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, uniform
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import least_squares


def normal_binary_search(low, high, **kwargs):
    bias = np.zeros(len(low))
    q = np.argmax(high - low)
    bias[q] = 1 - ((high + low) / 2)[q]
    return bias


def simultaneous_binary_search(low, high, **kwargs):
    # hyperrectangle relaxation    
    return 1 - (high + low) / 2


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

def mean_normal(low, high, constraints=None, real=None, error=None, **kwargs):
    # take the mean of the truncated normal distribution for each logit as a guess
    n = len(low)
    mu = 0.621457 
    sigma = 0.049797073
    # create a truncated norm for each logit
    truncated_norm = truncnorm(a=(low - mu) / sigma, b=(high - mu) / sigma, loc=mu, scale=sigma)
    # take the mean of the truncated norm for each logit
    r = truncated_norm.mean(axis=0)
    r[base_top_token] = 0
    print(r)
    return 1 - r
    

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
    r[base_top_token] = 0
    return r


def normal_perfect(low, high, constraints=None, real=None, error=None, precision=None, order=None, **kwargs):
    # plot mid and use low and high as error bars
    # introduce random error to real
    real_order = np.argsort(real)
    real = real[real_order]
    r = np.zeros(len(low))
    print('contraints')
    print(len(constraints))
    pos = order[(len(constraints) - 1) % len(low)] if order is not None else len(constraints) % len(low)
    print('position')
    print(pos)
    r[pos] -= precision if precision is not None else float('-inf')
    pos_r = np.where(real_order == pos)[0][0]
    if kwargs.get('plot', True):
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
    