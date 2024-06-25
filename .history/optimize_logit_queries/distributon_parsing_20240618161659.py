import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from run_attack import make_proper

data = np.load(
                "../distribution_logits/data/meta-llama/Llama-2-7b-hf-last_token_logits.pkl",
                allow_pickle=True,
            )

average_real = np.mean(data, axis=0)
## assumed width kinda random??
average_real = make_proper(average_real, assumed_width=40)

# Histogram and Gaussian fit
plt.hist(average_real, bins=100, density=True)
xmin, xmax = plt.xlim()
mu, sigma = stats.norm.fit(average_real)
print(mu, sigma)
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, "k", linewidth=2, label="Gaussian fit")

# skewed normal distribution
p_skewed = stats.skewnorm.pdf(x, a=5, scale=sigma)
plt.plot(x, p_skewed, "b", linewidth=2, label="Skewed Gaussian fit")

# now a bit sharper gaussian fit, without outliers
# Removing outliers for a sharper Gaussian fit
OUTLIERS = int(0.01 * len(average_real))
average_real_no_outliers = np.sort(average_real)[OUTLIERS:-OUTLIERS]
mu_no_outliers, sigma_no_outliers = stats.norm.fit(average_real_no_outliers)
print(f"Mu without outliers: {mu_no_outliers}, Sigma without outliers: {sigma_no_outliers}")
x_no_outliers = np.linspace(xmin, xmax, 100)
p_no_outliers = stats.norm.pdf(x_no_outliers, mu_no_outliers, sigma_no_outliers)
plt.plot(x_no_outliers, p_no_outliers, "r", linewidth=2, label="Gaussian fit without outliers")

# Calculate skewness and kurtosis
skewness = stats.skew(average_real)
kurtosis = stats.kurtosis(average_real)
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# Kernel Density Estimation
kde = stats.gaussian_kde(average_real)
p_kde = kde(x)
plt.plot(x, p_kde, 'r--', linewidth=2, label='KDE')

plt.legend()
plt.show()

# Sampling from the KDE
kde_samples = kde.resample(size=len(average_real))

# Save the KDE object
with open('kde_model.pkl', 'wb') as file:
    pickle.dump(kde, file)

# Plotting the KDE samples
plt.hist(kde_samples[0], bins=100, density=True, alpha=0.5, label='KDE Samples')
plt.plot(x, p_kde, 'r--', linewidth=2, label='KDE')
plt.legend()
plt.show()

# now we can sample from this distribution
samples = np.random.normal(mu, sigma, len(average_real))
plt.plot(samples)
plt.show()

# skewed normal distribution samples
skewed_samples = np.random.skewnorm(a=2, scale=sigma, size=len(average_real))
plt.plot(skewed_samples)
plt.show()

# # Sampling from the KDE
# kde_samples = kde.resample(size=len(average_real))

# # Plotting the KDE samples
# plt.hist(kde_samples, bins=100, density=True, alpha=0.5, label='KDE Samples')
# plt.plot(x, p_kde, 'r--', linewidth=2, label='KDE')
# plt.legend()
# plt.show()

average_real_sorted = np.sort(average_real)    
p_sorted = np.sort(samples)
p_sorted_skewed = np.sort(skewed_samples)

plt.plot(p_sorted)
plt.plot(p_sorted_skewed)
plt.plot(average_real_sorted)

plt.show()

