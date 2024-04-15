from matplotlib import pyplot as plt
import numpy as np
import os
from typing import Optional
from nptyping import Float, NDArray, Shape


def multivariate_normal(
	x: NDArray[Shape['D, *'], Float],
	mu: NDArray[Shape['D, *'], Float],
	cov: NDArray[Shape['D, D'], Float],
) -> float:
	det = np.linalg.det(cov)
	inv = np.linalg.inv(cov)
	d = len(x)
	denominator = np.sqrt((2 * np.pi) ** d * det)
	numerator = np.exp(-0.5 * (x - mu).T @ inv @ (x - mu))
	return numerator / denominator


def gmm(
	x: NDArray[Shape['D, *'], Float],
	phis: NDArray[Shape['K, *'], Float],
	mus: NDArray[Shape['K, D'], Float],
	covs: NDArray[Shape['K, D, D'], Float],
) -> float:
	K = len(phis)
	y = 0.0
	for k in range(K):
		phi, mu, cov = phis[k], mus[k, :], covs[k, :, :]
		y += phi * multivariate_normal(x, mu, cov)
	return y


def likelihood(
	xs: NDArray[Shape['N, D'], Float],
	phis: NDArray[Shape['K, *'], Float],
	mus: NDArray[Shape['K, D'], Float],
	covs: NDArray[Shape['K, D, D'], Float],
) -> float:
	eps = 1e-8
	L = 0
	N = len(xs)
	for x in xs:
		y = gmm(x, phis, mus, covs)
		L += np.log(y + eps)
	return L / N


class EMAlgorithm:
	def __init__(self,
		K: int,
		D: int,
		phis_init: Optional[NDArray[Shape['K, *'], Float]] = None,
		mus_init: Optional[NDArray[Shape['K, D'], Float]] = None,
		covs_init: Optional[NDArray[Shape['K, D, D'], Float]] = None,
	):
		self.K = K
		self.D = D

		if phis_init is None:
			self.phis = 0.5 * np.ones((K, ))
		else:
			self.phis = phis_init

		if mus_init is None:
			self.mus = np.zeros((K, D))
		else:
			self.mus = mus_init

		if covs_init is None:
			self.covs = np.array([np.eye(D) for _ in range(K)])
		else:
			self.covs = covs_init

	def e_step(
		self,
		xs: NDArray[Shape['N, D'], Float],
	) -> NDArray[Shape['N, K'], Float]:
		N = len(xs)
		K = self.K
		qs = np.zeros((N, K))
		for n in range(N):
			x = xs[n, :]
			for k in range(K):
				phi, mu, cov = self.phis[k], self.mus[k, :], self.covs[k, :, :]
				qs[n, k] = phi * multivariate_normal(x, mu, cov)
			qs[n, :] /= gmm(x, self.phis, self.mus, self.covs)
		return qs

	def m_step(
		self,
		xs: NDArray[Shape['N, D'], Float],
		qs: NDArray[Shape['N, K'], Float],
	):
		N = len(xs)
		K = self.K
		qs_sum = qs.sum(axis=0)
		for k in range(K):
			# 1. phis
			self.phis[k] = qs_sum[k] / N

			# 2. mus
			c = np.zeros((self.D, ))
			for n in range(N):
				c += qs[n, k] * xs[n, :]
			self.mus[k, :] = c / qs_sum[k]

			# 3. covs
			c = np.zeros((self.D, self.D))
			for n in range(N):
				z = xs[n, :] - self.mus[k, :]
				z = z[:, np.newaxis]
				c += qs[n, k] * z @ z.T
			self.covs[k, :, :] = c / qs_sum[k]

	def calculate(
		self,
		xs: NDArray[Shape['N, D'], Float],
		max_iter: int = 100,
		threshold: float = 1e-4,
	):
		current_likelihood = likelihood(xs, self.phis, self.mus, self.covs)

		for iter in range(max_iter):
			qs = self.e_step(xs)
			self.m_step(xs, qs)
			print(f'{current_likelihood:.3f}')

			next_likelihood = likelihood(xs, self.phis, self.mus, self.covs)
			diff = np.abs(next_likelihood - current_likelihood)
			if diff < threshold:
				break
			current_likelihood = next_likelihood

	def plot_contour(self):
		x = np.arange(1, 6, 0.1)
		y = np.arange(40, 100, 1)
		X, Y = np.meshgrid(x, y)
		Z = np.zeros_like(X)

		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				x = np.array([ X[i, j], Y[i, j] ])
				
				for k in range(len(self.mus)):
					phi, mu, cov = self.phis[k], self.mus[k, :], self.covs[k, :, :]
					Z[i, j] += phi * multivariate_normal(x, mu, cov)
		plt.contour(X, Y, Z)

	def generate(self, N: int) -> NDArray[Shape['N, K'], Float]:
		xs = np.zeros((N, self.D))
		for n in range(N):
			k = np.random.choice(self.K, p=self.phis)
			mu, cov = self.mus[k, :], self.covs[k, :, :]
			xs[n, :] = np.random.multivariate_normal(mu, cov)
		return xs



if __name__ == '__main__':
	path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
	xs = np.loadtxt(path)
	N, D = np.shape(xs)
	assert N == 272
	assert D == 2

	K = 2
	mus_init = np.array([[0., 50.], [0., 100.]])
	em = EMAlgorithm(K, D, mus_init=mus_init)
	em.calculate(xs)

	em.plot_contour()
	plt.scatter(xs[:, 0], xs[:, 1], label='original')
	new_xs = em.generate(100)
	plt.scatter(new_xs[:, 0], new_xs[:, 1], label='generated')
	plt.legend()

	plt.show()
