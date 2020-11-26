"""! @brief Variational Autoencoder generative modeling """

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from pipeline.components.model import Model
# from pipeline.components.model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import torchaudio

class BasicVAE(nn.Module):
    """! A basic implementation of a variational autoencoder.
    A variational autoencoder consists of two parts, the encoder and the decoder. It is a class of generative models. 
    Let 
    - \f$X\f$ be the data we are trying to model
    - \f$z\f$ be the latent variables
    - \f$P(X)\f$ be the probability distribution of the data
    - \f$P(X|z)\f$ be the proability distribution of the data given the latent variable

    Our goal here is to model the data, so we are marginalizing across our latent variables as shown below
    \f[
        P(X) = \int P(X|z)P(z)dz
    \f]

    With a VAE, we are inferring \f$P(z)\f$ usig \f$(P(z|X))\f$. To get that probability, we need to construct a probability such that we are projecting from our data space into our latent variable space. This is the encoder portion of our autoencoder.
    Let \f$Q(z|X)\f$ be our projection onto latent space. Now we use ELBO to define our objective function:
    \f[
        log(P(X)) - D_{KL}[Q(z|X)||P(z|X)] = \mathbf{E}[log(P(X|z))-D_{KL}[Q(z|X)||P(z)]]    
    \f]

    """ 
    def __init__(self, input_channels=6, **kwargs):
        """!
        The aim here is to have a simple Convolutional VAE with 2 convolutional blocks and two deconvolutional blocks. 

        """
        super(BasicVAE, self).__init__()
        latent_space_size = kwargs.get('latent_space_size', 8)
        encoder = kwargs.get('encoder', None)
        decoder = kwargs.get('decoder', None)
        kl_beta = kwargs.get('kl_beta', 1)
        intermed_size = kwargs.get('intermed', 420)
        ## Encoder
        if encoder:
            self.encoder = encoder()
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=12, kernel_size=3).double(),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(1896,3840).double(),
                nn.LeakyReLU(),
                ## Dense
                nn.Linear(3840, intermed_size).double(),
                nn.LeakyReLU()
            )
        ## mean and logvar
        self.mu_vec = nn.Linear(intermed_size,latent_space_size).double()# 594 calculated from above
        self.logvar_vec = nn.Linear(intermed_size, latent_space_size).double()
        ## Decoder 
        if decoder:
            self.decoder = decoder()
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_space_size,intermed_size).double(),
                nn.LeakyReLU(),
                nn.Linear(intermed_size,960).double(),
                nn.Unflatten(1,(6,160)),
            )
        weights_file = kwargs.get('weights_file', None)
        if weights_file:
            self.load_state_dict(torch.load(weights_file))

        ## Training parameters
        self.latent_size = latent_space_size
        self.kl_beta = kl_beta
    def reparameterization_trick(self, mu, log_var):
        """!
        Reparameterization trick to make latent space 
        @param mu  mean from the encoder's latent space
        @param log_var  log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        return sample
    
    def encode(self, data):
        return self.encoder(data)

    def decode(self, z):
        return self.decoder(z)
    def get_latent(self, data):
        ##Encoding
        x = self.encode(data)
        ##sampling
        mu = self.mu_vec(x)
        logvar = self.logvar_vec(x)
        z = self.reparameterization_trick(mu, logvar)
        return z
    def forward(self, data):
        """! forward propagation 
        @param data  sample data of input_size
        """
        ##Encoding
        x = self.encode(data)
        ##sampling
        mu = self.mu_vec(x)
        logvar = self.logvar_vec(x)
        z = self.reparameterization_trick(mu, logvar)

        ## decoding
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    def loss(self, reconstructed, original, mu, logvar):
        """!
        This is the loss function for our VAE. It consists of our reconstruction error plus the KL divergence

        """
        # start simple and just comput RMSE for reconstruction error
        # consider cross entropy with logit later
        reconstruction_error = torch.sqrt(functional.mse_loss(reconstructed, original))

        # KL Divergence -- We could replace this with some other loss as well
        kl_loss = self.kl_beta*(-0.5*torch.sum(1+ logvar - mu.pow(2) - logvar.exp()))
        return reconstruction_error, kl_loss #can be improved drastically
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3).double(),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1896,3840).double(),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            ## Dense
            nn.Linear(3840, 420).double(),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.encoder(x)
class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3).double(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=12, out_channels=18, kernel_size=3).double(),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=4, stride=2).double(),
            nn.Flatten(),
            nn.Linear(in_features=1386, out_features=693).double(),
            nn.BatchNorm1d(693).double(),
            nn.LeakyReLU(),
            nn.Linear(in_features=693, out_features=350).double(),
            nn.BatchNorm1d(350).double(),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.encoder(x)
class CNNLSTMEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CNNLSTMEncoder, self).__init__()
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3).double(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=12, out_channels=18, kernel_size=3).double(),
            nn.BatchNorm1d(18).double(),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2),
            nn.LSTM(input_size=78,
                    hidden_size=20,
                    num_layers=2,
                    batch_first=True).double(),
        )
        self.flat = nn.Flatten()
        self.linear = nn.Linear(360, 420).double()
    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.flat(x)
        x = self.linear(x)
        return x
class CNNLSTMEncoder2(nn.Module):
    def __init__(self, **kwargs):
        super(CNNLSTMEncoder2, self).__init__()
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3).double(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=12, out_channels=18, kernel_size=3).double(),
            nn.BatchNorm1d(18).double(),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(in_channels=18, out_channels=24, kernel_size=5).double(),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=24, out_channels=32, kernel_size=5).double(),
            nn.BatchNorm1d(32).double(),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2),
            nn.LSTM(input_size=35,
                    hidden_size=20,
                    num_layers=2,
                    batch_first=True).double(),
        )
        self.flat = nn.Flatten()
        self.linear = nn.Linear(640, 420).double()
    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.flat(x)
        x = self.linear(x)
        return x
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        # VGG without last layer
        self.initial_conv = nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=3,
        ).double()
        self.encoder = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1]).double()
        self.upsample = torch.nn.Upsample((64,64))
        self.flat = nn.Flatten()
        self.intermed = nn.Linear(25088, 1000).double()

    def forward(self, x):
        x = torchaudio.transforms.Spectrogram(n_fft = 160,
                                                win_length = 160,
                                                hop_length = 2,
                                                pad = 0,
                                                power = 2.0,
                                                normalized = False)(x)
        x = torchaudio.transforms.AmplitudeToDB(stype = 'power')(x)
        x = self.upsample(x)
        print(x.shape)
        x = self.initial_conv(x)
        x = self.encoder(x)
        x = self.flat(x)
        x = self.intermed(x)
        return x
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.decoder = nn.Sequential(
            nn.Unflatten(1,(4,64)),
            nn.Upsample(160),
            nn.Conv1d(
                in_channels=4,
                out_channels=6,
                kernel_size=1,
                padding_mode='reflect'
            ).double(),
            nn.ReLU(),
        )
    def forward(self, data):
        return self.decoder(data)
