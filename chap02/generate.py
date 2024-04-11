import numpy as np
from matplotlib import pyplot as plt
import os

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
assert xs.shape == (25000, )

mu = np.mean(xs)
sigma = np.std(xs)
assert np.isclose(mu, 172.7, atol=0.1, rtol=0.1)
assert np.isclose(sigma, 4.8, atol=0.1, rtol=0.1)

samples = np.random.normal(loc=mu, scale=sigma, size=(10000,))

plt.hist(xs, bins='auto', density=True, alpha=0.7, label='original')
plt.hist(samples, bins='auto', density=True, alpha=0.7, label='generated')
plt.xlabel('Height (cm)')
plt.ylabel('Probability density')
plt.legend()
plt.show()
