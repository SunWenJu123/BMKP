import torch.nn as nn
import torch
from torch import tensor
from torch.nn import functional as F
from backbone import He_init

class VAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims=None,
                 device='cpu') -> None:
        super(VAE, self).__init__()

        self.device = device

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

        # self.encoder.apply(He_init)
        # self.decoder.apply(He_init)
        # self.fc_mu.apply(He_init)
        # self.fc_var.apply(He_init)
        # self.decoder_input.apply(He_init)
        # self.final_layer.apply(He_init)


    def encode(self, input: tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # print("encoder", result.shape)
        result = torch.flatten(result, start_dim=1)
        # print(result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: tensor) -> tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (tensor) [B x D]
        :return: (tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # print("encoder", result.shape)
        result = result.view(z.shape[0], -1, 2, 2)
        # print(result.shape)
        result = self.decoder(result)
        result = self.final_layer(result)
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