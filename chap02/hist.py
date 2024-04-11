import numpy as np
from matplotlib import pyplot as plt
import os

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
assert xs.shape == (25000, )

plt.hist(xs, bins='auto', density=True)
plt.xlabel('Height (cm)')
plt.ylabel('Probability density')
plt.show()
