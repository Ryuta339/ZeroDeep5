import numpy as np
from numpy import random as rnd
from scipy import stats, linalg
import pytest

from em import *

@pytest.mark.parametrize('N', [2, 3, 5, 10])
def test_multivariate_normal(N: int):
	# random mean 
	mu = rnd.normal(size=(N, ))

	# random cov
	eigs = rnd.random(size=(N, ))
	U = stats.ortho_group.rvs(N)
	cov = U @ np.diag(eigs) @ U.T

	for _ in range(10):
		# random state
		x = rnd.normal(size=(N, ))
		expected = stats.multivariate_normal.pdf(x, mu, cov)
		actual = multivariate_normal(x, mu, cov)
		assert np.isclose(actual, expected)

@pytest.mark.parametrize('K', [2, 3, 5, 10])
def test_gmm(K: int):
	# random weight
	phis = rnd.random(size=(K,))
	phis /= phis.sum()

	# random mean
	mus = rnd.normal(loc=10, size=(K, 1))

	for _ in range(10):
		x = np.array([ rnd.normal(loc=10) ])
		expected = 0.0
		for k in range(K):
			expected += phis[k] * stats.multivariate_normal.pdf(x, mus[k], 1.0)
		actual = gmm(x, phis, mus, np.ones((K, 1, 1)))
		assert np.isclose(actual, expected)
