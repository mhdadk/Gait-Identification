import torch
from models.VAE_jeff_revA import BasicVAE
from dataloaders.CSVDataset_revC import CSVDataset
import numpy as np
import matplotlib.pyplot as plt
import os

window_length = 160
param_path = 'parameters/VAE_jeff.pt'
# net = VAE(window_length = 160,
#           param_path = param_path)
net = BasicVAE(weights_file = param_path,
                latent_space_size = 256).to(torch.float64)
# net.load_state_dict(torch.load('best_param.pt'))
net.eval()

data_dir = '../../data/projF/train'
test_path = os.path.join(data_dir,'x_test.csv')
dataset = CSVDataset(test_path)

x = dataset[21].unsqueeze(dim=0)

x_hat = net(x)[0]

# x_hat = torch.sigmoid(x_hat)

x = x.squeeze().numpy()
x_hat = x_hat.detach().squeeze().numpy()

t = np.arange(x.shape[1])
plt.plot(t,x[5,:],'b',t,x_hat[5,:],'r')