import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal

n_samples = 100
mu1, sigma1 = -5, 1.6 
mu2, sigma2 = 2, 3.1 
x1 = np.random.normal(loc = mu1, scale = np.sqrt(sigma1), size = n_samples)
x2 = np.random.normal(loc = mu2, scale = np.sqrt(sigma2), size = n_samples)
X = np.concatenate((x1,x2))

def plot_pdf(mu,sigma,label,color,alpha=.8,density=True):
    """
    Plot 1-D data and its PDF curve.
    """
    # Compute the mean and standard deviation of the data
    # Plot the data
    X = norm.rvs(mu, sigma, size=1000)
    plt.hist(X, bins=50, density=density, alpha=alpha, color=color)
    # Plot the PDF
    x = np.linspace(X.min(), X.max(), 1000)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, color, lw=linewidth)
    
linewidth = 5
"""
plot_pdf(mu1,sigma1,label=r"$mu={}  ;  sigma={}$".format(mu1,sigma1),color="#6cafe9")
plot_pdf(mu2,sigma2,label=r"$mu={}  ;  sigma={}$".format(mu2,sigma2),color="#5b86ff")
plt.legend()
plt.show()
"""




# Set a random seed for reproducible points
np.random.seed(42)

# 1. Create a 2D coordinate grid for the contour lines
x, y = np.meshgrid(np.linspace(-5, 6, 200), np.linspace(-5, 6, 200))
pos = np.dstack((x, y))

# 2. Define the two 2D Gaussian distributions
rv1 = multivariate_normal(mean=[-1.0, -1.0], cov=[[1.0, 0.5], [0.5, 1.0]])
rv2 = multivariate_normal(mean=[2.0, 1.5], cov=[[1.5, -0.7], [-0.7, 1.2]])

# 3. Generate random sample points from each distribution
samples1 = rv1.rvs(size=800)
samples2 = rv2.rvs(size=800)

# 4. Plot scatter points and overlay PDF contours
fig, ax = plt.subplots(figsize=(7, 6))

# Scatter plots (alpha makes overlapping points easier to see)
ax.scatter(samples1[:, 0], samples1[:, 1], color="#6cafe9", alpha=0.72, s=22)
ax.scatter(samples2[:, 0], samples2[:, 1], color="#5b86ff", alpha=0.72, s=22)

# Contour lines for the underlying densities
ax.contour(x, y, rv1.pdf(pos), levels=5, colors="#6cafe9", linewidths=2.6)
ax.contour(x, y, rv2.pdf(pos), levels=5, colors="#5b86ff", linewidths=2.6)

ax.legend()
plt.show()