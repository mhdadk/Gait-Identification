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
        
        # conv layers for mu and logvar in reparameterization trick
        
        self.conv_mu = torch.nn.Conv1d(in_channels = 1,
                                       out_channels = 1,
                                       kernel_size = 745,
                                       stride = 1,
                                       padding = 0)

        self.conv_logvar = torch.nn.Conv1d(in_channels = 1,
                                           out_channels = 1,
                                           kernel_size = 745,
                                           stride = 1,
                                           padding = 0)
        
        # decoder
        
        linear1 = torch.nn.Linear(in_features = 64,
                                  out_features = 160)
        
        conv2 = torch.nn.Conv1d(in_channels = 4,
                                out_channels = 6,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                padding_mode = 'reflect')
        
        self.decoder = torch.nn.Sequential(linear1,
                                           conv2)
        
    def forward(self,_x):
        
        # input _x has dimensions batch_size x num_channels x window_length
        
        # convert to batch_size x 3 x 224 x 224 tensor
        
        x = self.preprocessing(_x)
        
        # encode. Unsqueeze for conv1d later
        
        x = self.encoder(x).unsqueeze(dim = 1)
        
        # reparameterization trick
        
        self.mu = self.conv_mu(x)
        self.logvar = self.conv_logvar(x)
        
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