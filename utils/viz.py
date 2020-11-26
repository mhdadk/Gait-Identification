import torch
import os

from VAE import BasicVAE
from CSVDataset import CSVDataset

net = BasicVAE(input_shape = (160,6),
               latent_space_size = 3,
               kl_beta = 0.0)

net.load_state_dict(torch.load('3d_vae.pt'))

data_dir = '../../../data/projF/train'
val_x_path = os.path.join(data_dir,'x_val.csv')
val_y_path = os.path.join(data_dir,'y_val.csv')
test_path = os.path.join(data_dir,'x_test.csv')

dataset = CSVDataset(x_csv_path = val_x_path,
                     y_csv_path = val_y_path,
                     window_length = 160,
                     window_overlap = 0.4)

dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = 1,
                                         shuffle = False)

x,y = dataset[0]

x = x.transpose(-2,-1).unsqueeze(dim=0)
recon,mu,logvar = net(x)

import matplotlib.pyplot as plt

t = list(range(x.shape[2]))
x = x.squeeze().detach().numpy()
recon = recon.squeeze().detach().numpy()
plt.plot(t,x[0,:],'b',t,recon[0,:],'r')