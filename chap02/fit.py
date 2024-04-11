import numpy as np
from matplotlib import pyplot as plt
import os

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
assert xs.shape == (25000, )

plt.hist(xs, bins='auto', density=True)

mu = np.mean(xs)
sigma = np.std(xs)

def normal(x: float, mu: float = 0, sigma: float = 0):
	denominator = np.sqrt(2 * np.pi) * sigma
	numerator = np.exp(-0.5 * (x - mu)**2 / sigma**2)
	return numerator / denominator

x = np.linspace(np.min(xs), np.max(xs))
plt.plot(x, normal(x, mu, sigma))

plt.xlabel('Height (cm)')
plt.ylabel('Probability density')
plt.show()


