import torch
import torchaudio
import torchvision

batch_size = 4
num_channels = 6
window_length = 160
x = torch.randn(batch_size,num_channels,window_length)

# zero mean each channel to avoid outliers in spectrograms

for i in range(x.shape[1]):
    temp = x[:,i,:].transpose(0,1) - torch.mean(x[:,i,:],dim=1)
    x[:,i,:] = temp.transpose(0,1)

# log spectrogram

spec = torchaudio.transforms.Spectrogram(n_fft = window_length,
                                         win_length = window_length,
                                         hop_length = 2,
                                         pad = 0,
                                         power = 2.0,
                                         normalized = False)(x)
log_spec = torchaudio.transforms.AmplitudeToDB(stype = 'power')(spec)

# transform to 3 x 224 x 224

upsample = torch.nn.Upsample(size = (224,224),
                             mode = 'nearest')

y1 = upsample(log_spec)

conv1 = torch.nn.Conv2d(in_channels = 6,
                        out_channels = 3,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,
                        padding_mode = 'reflect')

y2 = conv1(y1)

# VGG encoder

encoder = torchvision.models.resnet18(pretrained=False)

# normalize for imagenet stats

# y3 = torchvision.transforms.functional.normalize(y2,
#                                           [0.485, 0.456, 0.406],
#                                           [0.229, 0.224, 0.225])

y3 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])(y2)

# unsqueeze for conv1d

y4 = encoder(y3).unsqueeze(dim=1)

# reparameterization trick

conv_mu = torch.nn.Conv1d(in_channels = 1,
                          out_channels = 1,
                          kernel_size = 745,
                          stride = 1,
                          padding = 0,
                          padding_mode = 'reflect')

mu = conv_mu(y4)

conv_logvar = torch.nn.Conv1d(in_channels = 1,
                              out_channels = 1,
                              kernel_size = 745,
                              stride = 1,
                              padding = 0,
                              padding_mode = 'reflect')

logvar = conv_logvar(y4)

std = torch.exp(0.5*logvar)
eps = torch.randn_like(std)

y5 = mu + eps*std

# decoder

y6 = torch.reshape(y5,(batch_size,4,64))

final_linear = torch.nn.Linear(in_features = 64,
                               out_features = 160)

y7 = final_linear(y6)

# upsample = torch.nn.Upsample(size = 160,
#                              mode = 'nearest')

# y7 = upsample(y6)

# final_linear = torch.nn.Linear(in_features = 160,
#                                out_features = 160)

conv2 = torch.nn.Conv1d(in_channels = 4,
                        out_channels = 6,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,
                        padding_mode = 'reflect')

y8 = conv2(y7)
