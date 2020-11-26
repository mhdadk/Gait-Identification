import torch

batch_size = 32
x = torch.randn(batch_size,1,160,6)
x_hat = torch.randn(batch_size,1,160,6)

mu = torch.randn(batch_size,1,3)
std = torch.randn(batch_size,1,3)
logvar = torch.randn(batch_size,1,3)
beta = 1

MSE = torch.mean((x - x_hat)**2)  

# see Appendix B from VAE paper:
# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
# https://arxiv.org/abs/1312.6114
# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=2)
KLD = beta*torch.mean(KLD)
    
