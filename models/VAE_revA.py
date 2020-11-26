import torch

class VAE(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        activation = torch.nn.ReLU()
        
        conv1 = torch.nn.Conv2d(in_channels = 1,
                                out_channels = 2,
                                kernel_size = (40,2))
        
        conv2 = torch.nn.Conv2d(in_channels = 2,
                                out_channels = 3,
                                kernel_size = (40,2))
        
        conv3 = torch.nn.Conv2d(in_channels = 3,
                                out_channels = 4,
                                kernel_size = (40,2))
        
        conv4 = torch.nn.Conv2d(in_channels = 4,
                                out_channels = 12,
                                kernel_size = (43,3))
        
        self.encoder = torch.nn.Sequential(conv1,activation,
                                           conv2,activation,
                                           conv3,activation,
                                           conv4,activation)
        
        deconv1 = torch.nn.ConvTranspose2d(in_channels = 1,
                                           out_channels = 1,
                                           kernel_size = (2,2))
        
        deconv2 = torch.nn.ConvTranspose2d(in_channels = 1,
                                           out_channels = 2,
                                           kernel_size = (2,2))
        
        deconv3 = torch.nn.ConvTranspose2d(in_channels = 2,
                                           out_channels = 2,
                                           kernel_size = (2,2))
        
        deconv4 = torch.nn.ConvTranspose2d(in_channels = 2,
                                           out_channels = 3,
                                           kernel_size = (4,4))
        
        deconv5 = torch.nn.ConvTranspose2d(in_channels = 3,
                                           out_channels = 6,
                                           kernel_size = (8,4))
        
        self.decoder = torch.nn.Sequential(deconv1,activation,
                                           deconv2,activation,
                                           deconv3,activation,
                                           deconv4,activation,
                                           deconv5,activation)
        
        self.linear_tan = torch.nn.Linear(in_features = 3,
                                          out_features = 3)
        
        self.linear_mean = torch.nn.Linear(in_features = 3,
                                           out_features = 3)
        
        self.linear_logvar = torch.nn.Linear(in_features = 3,
                                             out_features = 3)
        
    def forward(self,_x):
        
        # encode
        
        x = self.encoder(_x)
        
        # flatten to batch of 12-dimensional row vectors
        
        x = torch.reshape(x,(_x.shape[0],1,12))
        
        # maxpool to get batch of 3-dimensional vectors
        
        x = torch.nn.MaxPool1d(kernel_size = 4)(x)
        
        # get mean and square root of covariance matrix. See section C.2
        # in original VAE paper by Kingma et al. for details
        
        h = torch.nn.Tanh()(self.linear_tan(x))
        self.mean = self.linear_mean(h)
        self.logvar = self.linear_logvar(h)
        
        # reparameterization trick
        
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        z = (torch.mul(eps,std) + self.mean).unsqueeze(dim=-1)
        
        # decode
        
        y = self.decoder(z)
        
        # reshape to compare to input
        
        y = torch.reshape(y,_x.shape)
        
        return y
    
if __name__ == '__main__':
    
    vae = VAE()
    
    batch_size = 64
    
    x = torch.ones(batch_size,1,160,6)
    
    y = vae(x)