import torch
import torchaudio
import torchvision

class VAE(torch.nn.Module):
    
    def __init__(self,window_length):
        
        super().__init__()
        
        # preprocessing
        
        # log spectrogram

        spec = torchaudio.transforms.Spectrogram(n_fft = window_length,
                                                 win_length = window_length,
                                                 hop_length = 2,
                                                 pad = 0,
                                                 power = 2.0,
                                                 normalized = False)
        log_spec = torchaudio.transforms.AmplitudeToDB(stype = 'power')
        
        # transform to 3 x 224 x 224
        
        upsample1 = torch.nn.Upsample(size = (224,224),
                                      mode = 'nearest')
        
        conv1 = torch.nn.Conv2d(in_channels = 6,
                                out_channels = 3,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                padding_mode = 'reflect')
        
        self.preprocessing = torch.nn.Sequential(spec,
                                                 log_spec,
                                                 upsample1,
                                                 conv1)
        
        # pre-trained encoder with normalization
        
        # normalization = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
        #                                                  std = [0.229, 0.224, 0.225])
        
        self.encoder = torchvision.models.resnet18(pretrained = True,
                                                   progress = True)
        
        # self.encoder = torch.nn.Sequential(normalization,
        #                                    net)
        
        # linear layers for mu and logvar in reparameterization trick
        
        self.linear_mu = torch.nn.Linear(in_features = 1000,
                                         out_features = 256)

        self.linear_logvar = torch.nn.Linear(in_features = 1000,
                                             out_features = 256)
        
        # decoder
        
        upsample2 = torch.nn.Upsample(size = window_length,
                                      mode = 'nearest')
        
        conv2 = torch.nn.Conv1d(in_channels = 4,
                                out_channels = 6,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                padding_mode = 'reflect')
        
        activation = torch.nn.ReLU()
        
        self.decoder = torch.nn.Sequential(upsample2,
                                           conv2,
                                           activation)
        
    def forward(self,_x):
        
        # input _x has dimensions batch_size x num_channels x window_length
        
        # convert to batch_size x 3 x 224 x 224 tensor
        
        x = self.preprocessing(_x)
        
        # encode
        
        x = self.encoder(x)
        
        # reparameterization trick
        
        self.mu = self.linear_mu(x)
        self.logvar = self.linear_logvar(x)
        
        std = torch.exp(0.5 * self.logvar)
        epsilon = torch.randn_like(std)
        
        z = self.mu + epsilon * std
        
        # decode
        
        y = torch.reshape(z,(_x.shape[0],4,64))
        
        y = self.decoder(y)
        
        return y

if __name__ == '__main__':
    
    net = VAE(window_length = 160)
    
    batch_size = 4
    num_channels = 6
    window_length = 160
    x = torch.randn(batch_size,num_channels,window_length)
    
    y = net(x)