import torch

class VAE(torch.nn.Module):
    
    def __init__(self,window_length,param_path):
        
        super().__init__()
        
        # load parameters
        
        params = torch.load(param_path)
        
        # activation function
        
        activation = torch.nn.LeakyReLU()
        
        # utils
        
        flatten = torch.nn.Flatten()
        unflatten = torch.nn.Unflatten(1,(6,window_length))
        
        # encoder ########################################################
        
        # first convolutional layer
        
        conv1 = torch.nn.Conv1d(in_channels = 6,
                                out_channels = 12,
                                kernel_size = 3).double()
        
        conv1.weight = torch.nn.Parameter(params['encoder.0.weight'])
        conv1.bias = torch.nn.Parameter(params['encoder.0.bias'])
        
        # first fully-connected layer
        
        linear1 = torch.nn.Linear(in_features = 1896,
                                  out_features = 3840).double()
        
        linear1.weight = torch.nn.Parameter(params['encoder.3.weight'])
        linear1.bias = torch.nn.Parameter(params['encoder.3.bias'])
        
        # second fully-connected layer
        
        linear2 = torch.nn.Linear(in_features = 3840,
                                  out_features = 420).double()
        
        linear2.weight = torch.nn.Parameter(params['encoder.5.weight'])
        linear2.bias = torch.nn.Parameter(params['encoder.5.bias'])
        
        self.encoder = torch.nn.Sequential(conv1,
                                           activation,
                                           flatten,
                                           linear1,
                                           activation,
                                           linear2,
                                           activation)
        
        # latent space ###################################################
        
        self.linear_mu = torch.nn.Linear(in_features = 420,
                                         out_features = 256).double()
        
        self.linear_mu.weight = torch.nn.Parameter(params['mu_vec.weight'])
        self.linear_mu.bias = torch.nn.Parameter(params['mu_vec.bias'])
        
        self.linear_logvar = torch.nn.Linear(in_features = 420,
                                             out_features = 256)
        
        self.linear_logvar.weight = torch.nn.Parameter(params['logvar_vec.weight'])
        self.linear_logvar.bias = torch.nn.Parameter(params['logvar_vec.bias'])
        
        # decoder ########################################################
        
        # first fully-connected layer
        
        linear3 = torch.nn.Linear(in_features = 256,
                                  out_features = 420).double()
        
        linear3.weight = torch.nn.Parameter(params['decoder.0.weight'])
        linear3.bias = torch.nn.Parameter(params['decoder.0.bias'])
        
        # second fully-connected layer
        
        linear4 = torch.nn.Linear(in_features = 420,
                                  out_features = 960).double()
        
        linear4.weight = torch.nn.Parameter(params['decoder.2.weight'])
        linear4.bias = torch.nn.Parameter(params['decoder.2.bias'])
        
        self.decoder = torch.nn.Sequential(linear3,
                                           activation,
                                           linear4,
                                           unflatten)
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        
        return z
    
    def sample_z_given_x(self,x):
        
        x = self.encoder(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        z = self.reparameterize(mu,logvar)
        
        return z
    
    def forward(self,x):
        
        z = self.sample_z_given_x(x)
        x_hat = self.decoder(z)
        
        return x_hat

if __name__ == '__main__':
    
    net = VAE(window_length = 160,
              param_path = '../parameters/VAE_jeff.pt')
    
    batch_size = 4
    num_channels = 6
    window_length = 160
    x = torch.randn(batch_size,num_channels,window_length).double()
    
    y = net(x)
    z = net.sample_z_given_x(x)