import torch.nn as nn
import torch
from torch import tensor
from torch.nn import functional as F

class VAE_MLP(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dims=None,
                 input_dim=28*28,
                 is_mnist=True,
                 device='cpu') -> None:
        super(VAE_MLP, self).__init__()

        self.device = device

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.is_mnist = is_mnist

        if hidden_dims is None:
            hidden_dims = [100, 100]

        # Build Encoder

        modules = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        hidden_dims.reverse()
        modules = []
        in_dim = self.latent_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_dim = h_dim

        self.decoder = nn.Sequential(*modules)

        # print(hidden_dims)
        if is_mnist:
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.input_dim),
                nn.Tanh(),
            )
        else:
            self.final_layer = nn.Linear(hidden_dims[-1], self.input_dim)

    def encode(self, input: tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (tensor) Input tensor to encoder [N x C x H x W]
        :return: (tensor) List of latent codes
        """
        input = input.view(input.shape[0], -1)

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: tensor) -> tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param isMnist:
        :param z: (tensor) [B x D]
        :return: (tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        if self.is_mnist:
            result = result.view(-1, 1, 28, 28)
        else:
            result = result.view(-1, self.input_dim)
        return result

    def reparameterize(self, mu: tensor, logvar: tensor) -> tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return output, input, mu, log_var

    def sample(self, num_samples:int) -> tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(self.device)

        samples = self.decode(z)
        return samples

    def loss_function(self, input, recons, mu, log_var, kld_weight) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}