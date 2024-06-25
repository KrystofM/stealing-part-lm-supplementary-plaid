import numpy as np
from scipy.stats import truncnorm, spearmanr, kendalltau
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import pdist, squareform

def distance_correlation(X, Y):
    X = np.atleast_2d(X).T
    Y = np.atleast_2d(Y).T
    n = X.shape[0]
    a = squareform(pdist(X, 'euclidean'))
    b = squareform(pdist(Y, 'euclidean'))
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov = np.sqrt((A * B).sum() / (n * n))
    dvarx = np.sqrt((A * A).sum() / (n * n))
    dvary = np.sqrt((B * B).sum() / (n * n))
    return dcov / np.sqrt(dvarx * dvary)

def sample_truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def compute_correlations(mean1, std1, lower1, upper1, mean2, std2, lower2, upper2, size=10000):
    # Sample from the two truncated normal distributions
    X = sample_truncated_normal(mean1, std1, lower1, upper1, size)
    Y = sample_truncated_normal(mean2, std2, lower2, upper2, size)
    
    # Compute Pearson correlation
    pearson_corr = np.corrcoef(X, Y)[0, 1]
    
    # Compute Spearman's rank correlation
    spearman_corr, _ = spearmanr(X, Y)
    
    # Compute Kendall's Tau
    kendall_tau, _ = kendalltau(X, Y)
    
    # Compute Mutual Information
    mutual_info = mutual_info_score(np.digitize(X, bins=np.histogram_bin_edges(X, bins='auto')),
                                    np.digitize(Y, bins=np.histogram_bin_edges(Y, bins='auto')))
    
    # Compute Distance Correlation
    dist_corr = distance_correlation(X, Y)
    
    return {
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
        'Kendall': kendall_tau,
        'Mutual Information': mutual_info,
        'Distance Correlation': dist_corr
    }

# Example usage
mean1, std1, lower1, upper1 = 0, 1, -1, 1
mean2, std2, lower2, upper2 = 0, 1, -1, 1
correlations = compute_correlations(mean1, std1, lower1, upper1, mean2, std2, lower2, upper2)
print(correlations)