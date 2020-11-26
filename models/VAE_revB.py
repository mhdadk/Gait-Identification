"""! @brief Variational Autoencoder generative modeling """

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np

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
    def __init__(self, input_shape=(160,6), latent_space_size=3, kl_beta=0.0):
        """!
        The aim here is to have a simple Convolutional VAE with 2 convolutional blocks and two deconvolutional blocks. 

        """
        super(BasicVAE, self).__init__()

        ## Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3).double(),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1896,3840).double(),
            nn.LeakyReLU(),
            ## Dense
            nn.Linear(3840, 420).double(),
            nn.LeakyReLU()
        )

        ## mean and logvar
        self.mu_vec = nn.Linear(420,latent_space_size).double()# 594 calculated from above
        self.logvar_vec = nn.Linear(420, latent_space_size).double()

        ## Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_size,420).double(),
            nn.LeakyReLU(),
            nn.Linear(420,960).double(),
            nn.Unflatten(1,(6,160)),
        )
        
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        #self.encoder.apply(init_weights)
        #self.decoder.apply(init_weights)
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

class BasicVAETrainer():
    def __init__(self, vae, params, optimizer=optim.SGD):
        """!
        Class that manages training for BasicVAE
        @params vae  vae model being trained
        @params optimizer  type of optimizer to use
        @params params  dictionary containing 'training_parameters' and 'optimizer_parameters'
        """
        self.vae = vae
        optimizer_params = params['optimizer_parameters']
        train_params = params['training_parameters']

        self.optimizer = optimizer(params=vae.parameters(), **optimizer_params)
        self.epochs = train_params.get('epochs', 50)
        self.batch_size = train_params.get('batch_size', 86)

    def train_step(self, data): #training step
        """! Executes one training step and returns the loss.
        @param data  data used to compute loss
        """
        reconstruction, mu, logvar = self.vae.forward(data)
        recon_loss, kl_loss = self.vae.loss(reconstruction, data, mu, logvar)
        self.optimizer.zero_grad() # sets gradient to zero -- needed to prevent accumulation
        loss = recon_loss + kl_loss
        loss.backward() # loss + computational graph 
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 0.5)
        self.optimizer.step() # Updates weights 
        return recon_loss.detach().item(), kl_loss.detach().item()
        
    def train(self, training_data, validation_data=None):
        training_history, validation_history = [], []
        train_loader = torch.utils.data.DataLoader(training_data, self.batch_size, True)
        kl_beta_schedule = list(np.arange(4.1e-8,5e-6,4.1e-8))
        for epoch in range(self.epochs): # run epochs
            tloss_mse, tloss_kl = 0,0
            ## Training Stage 
            for batch in train_loader:
                del_tloss_mse, del_tloss_kl = self.train_step(batch)
                tloss_mse += del_tloss_mse
                tloss_kl += del_tloss_kl
            #print(f'batch data shape is {batch.shape} and params {batch_start}, {batch_start + self.batch_size}')
            self.vae.kl_beta = kl_beta_schedule.pop(0)
            tloss_mse /= self.batch_size
            tloss_kl /= self.batch_size
            training_history.append((tloss_mse + tloss_kl))
            ## Validation Stage
            if validation_data is not None:
                reconstructions, mu, logvar = self.vae.forward(validation_data)
                vloss,_ = self.vae.loss(reconstructions, validation_data, mu, logvar) # only care about reconstruction loss for validation
                validation_history.append(vloss.detach().item())
                print(f'epoch {epoch} -- train_mse : {tloss_mse:.4f} -- train_kl : {tloss_kl:.4f} -- val_loss : {vloss:.4f}')
            else:
                print(f'epoch {epoch} -- training_loss : {tloss_mse:.4f} -- train_kl : {tloss_kl:.4f}')
        return training_history, validation_history

