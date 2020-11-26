import torch

x = torch.ones(32,1,160,6)

conv1 = torch.nn.Conv2d(in_channels = 1,
                        out_channels = 2,
                        kernel_size = (40,2))

y1 = conv1(x)

conv2 = torch.nn.Conv2d(in_channels = 2,
                        out_channels = 3,
                        kernel_size = (40,2))

y2 = conv2(y1)

conv3 = torch.nn.Conv2d(in_channels = 3,
                        out_channels = 4,
                        kernel_size = (40,2))

y3 = conv3(y2)

conv4 = torch.nn.Conv2d(in_channels = 4,
                        out_channels = 12,
                        kernel_size = (43,3))

y4 = conv4(y3).reshape((32,1,12))

y4 = torch.nn.MaxPool1d(kernel_size = 4)(y4)

mean = y4[:,:3].unsqueeze(dim=-1)
std = y4[:,3:].reshape((32,3,3)) # square this to get covariance matrix
epsilon = torch.randn(32,3,1)

z = (torch.matmul(std,epsilon) + mean).unsqueeze(dim=1)

deconv1 = torch.nn.ConvTranspose2d(in_channels = 1,
                                   out_channels = 1,
                                   kernel_size = (2,2))

y5 = deconv1(z)

deconv2 = torch.nn.ConvTranspose2d(in_channels = 1,
                                   out_channels = 2,
                                   kernel_size = (2,2))

y6 = deconv2(y5)

deconv3 = torch.nn.ConvTranspose2d(in_channels = 2,
                                   out_channels = 2,
                                   kernel_size = (2,2))

y7 = deconv3(y6)

deconv4 = torch.nn.ConvTranspose2d(in_channels = 2,
                                   out_channels = 3,
                                   kernel_size = (4,4))

y8 = deconv4(y7)

deconv5 = torch.nn.ConvTranspose2d(in_channels = 3,
                                   out_channels = 6,
                                   kernel_size = (8,4))

y9 = deconv5(y8).reshape((32,1,160,6))
