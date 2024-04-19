from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data
import torchvision
from torchvision import datasets, transforms
from typing import Tuple

class Encoder(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dim: int,
		latent_dim: int,
	):
		super().__init__()
		self.linear = nn.Linear(input_dim, hidden_dim)
		self.linear_mu = nn.Linear(hidden_dim, latent_dim)
		self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

	def forward(
		self,
		x: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.linear(x)
		h = F.relu(h)
		mu = self.linear_mu(h)
		logvar = self.linear_logvar(h)
		sigma = torch.exp(0.5 * logvar)
		return mu, sigma


class Decoder(nn.Module):
	def __init__(
		self,
		latent_dim: int,
		hidden_dim: int,
		output_dim: int,
	):
		super().__init__()
		self.linear1 = nn.Linear(latent_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, output_dim)

	def forward(
		self,
		z: torch.Tensor,
	) -> torch.Tensor:
		h = self.linear1(z)
		h = F.relu(h)
		h = self.linear2(h)
		x_hat = F.sigmoid(h)
		return x_hat


def reparameterize(
	mu: torch.Tensor,
	sigma: torch.Tensor,
) -> torch.Tensor:
	eps = torch.randn_like(sigma)
	z = mu + eps * sigma
	return z


class VAE(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dim: int,
		latent_dim: int,
	):
		super().__init__()
		self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
		self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

	def get_loss(
		self,
		x: torch.Tensor,
	):
		mu, sigma = self.encoder(x)
		z = reparameterize(mu, sigma)
		x_hat = self.decoder(z)

		batch_size = len(x)
		L1 = F.mse_loss(x_hat, x, reduction='sum')
		L2 = -torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
		return (L1 + L2) / batch_size


def learn(
	model: nn.Module,
	optimizer: optim.Optimizer,
	dataloader: data.DataLoader,
	epochs: int = 20,
):
	losses = []

	for epoch in range(epochs):
		loss_sum = 0.0
		cnt = 0
		
		for x, label in dataloader:
			optimizer.zero_grad()
			loss = model.get_loss(x)
			loss.backward()
			optimizer.step()

			loss_sum += loss.item()
			cnt += 1

		loss_avg = loss_sum / cnt
		losses.append(loss_avg)
	
	return losses


if __name__ == '__main__':
	input_dim = 784
	hidden_dim = 200
	latent_dim = 20
	learning_rate = 3e-4
	batch_size = 32

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(torch.flatten)
	])
	dataset = datasets.MNIST(
		root='./data',
		train=True,
		download=True,
		transform=transform
	)
	dataloader = data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True
	)

	model = VAE(input_dim, hidden_dim, latent_dim)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	losses = learn(model, optimizer, dataloader)

	with torch.no_grad():
		sample_size = 64
		z = torch.randn(sample_size, latent_dim)
		x = model.decoder(z)
		generated_images = x.view(sample_size, 1, 28, 28)

	grid_img = torchvision.utils.make_grid(
		generated_images,
		nrow=8,
		padding=2,
		normalize=True
	)

	plt.imshow(grid_img.permute(1, 2, 0))
	plt.axis("off")
	plt.show()