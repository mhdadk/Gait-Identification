import torch

from models.VAE_revC import VAE
from dataloaders.CSVDataset_revB import CSVDataset

import os
import copy
import time

def train_batch(x,beta,device,net,loss_func,optimizer):
    
    # zero the accumulated parameter gradients
    
    optimizer.zero_grad()
    
    # outputs of net for batch input
    
    x_hat = net(x)
    
    # compute loss
    
    MSE,KLD = loss_func(x,x_hat,net.mu,net.logvar,beta)
    loss = MSE - KLD
    
    # compute loss gradients with respect to parameters
    
    loss.backward()
    
    # update parameters according to optimizer
    
    optimizer.step()
    
    return MSE.item(),KLD.item()

def train_epoch(net,beta,dataloader,device,loss_func,optimizer):
    
    # put net in training mode
    
    net.train()
    print('Training...')
    
    # to compute total training loss over entire epoch
    
    total_mse_loss = 0
    total_kld_loss = 0
    
    for i,x in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # move to GPU
    
        x = x.to(device)
        
        # train over the batch
    
        MSE,KLD = train_batch(x,beta,device,net,loss_func,optimizer)
        
        # record total loss
        
        total_mse_loss += MSE
        total_kld_loss += KLD
    
    mse_loss = total_mse_loss / len(dataloader.dataset)
    kld_loss = total_kld_loss / len(dataloader.dataset)
    
    return mse_loss,kld_loss

def val_batch(x,beta,device,net,loss_func):
    
    # outputs of net for batch input
    
    with torch.no_grad():
    
        x_hat = net(x)
    
        MSE,KLD = loss_func(x,x_hat,net.mu,net.logvar,beta)
    
    return MSE.item(),KLD.item()

def val_epoch(net,beta,dataloader,device,loss_func):
    
    # put net in testing mode
                  
    net.eval()
    print('\nValidating...')
    
    # to compute total validation loss over entire epoch
    
    total_mse_loss = 0
    total_kld_loss = 0
    
    for i,x in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # move to GPU
        
        x = x.to(device)
        
        # validate over the batch
        
        MSE,KLD = val_batch(x,beta,device,net,loss_func)
        
        # record total loss
        
        total_mse_loss += MSE
        total_kld_loss += KLD
    
    mse_loss = total_mse_loss / len(dataloader.dataset)
    kld_loss = total_kld_loss / len(dataloader.dataset)
    
    return mse_loss,kld_loss

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = False #torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize VAE net and convert it to torch.float64 dtype to match
# type of input tensors

window_length = 160
net = VAE(window_length = window_length).to(device).to(torch.float64)

# initialize datasets and dataloaders

data_dir = '../../data/projF/train'
train_path = os.path.join(data_dir,'x_train.csv')
val_path = os.path.join(data_dir,'x_val.csv')
test_path = os.path.join(data_dir,'x_test.csv')
dataloaders = {}

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# batch sizes for training, validation, and testing

train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

for mode,path,batch_size in [('train',train_path,train_batch_size),
                             ('val',val_path,val_batch_size),
                             ('test',test_path,test_batch_size)]:
    
    dataset = CSVDataset(path)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = False,
                               **dl_config)

# initialize loss function

def loss_func(x,x_hat,mu,logvar,beta):
    
    # need to use reduction = 'mean' because .backward() can only use scalars
    
    MSE = torch.nn.functional.binary_cross_entropy_with_logits(x_hat,
                                                               x,
                                                               reduction='mean')
    # MSE = torch.mean((x - x_hat)**2)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    # sum over dimensions of z
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=-1)
    
    # multiply by weight and average over batch
    
    KLD = beta*torch.mean(KLD)
    
    return MSE,KLD

# initialize optimizer. Must put net parameters on GPU before this step.
# set different learning rates and weight decay for different parameters

preprocessing_params = {'params':net.preprocessing.parameters(),
                        'lr':1e-4,
                        'weight_decay':1e-3}

encoder_params = {'params':net.encoder.parameters(),
                  'lr':1e-9,
                  'weight_decay':0}

mu_params = {'params':net.conv_mu.parameters(),
             'lr':1e-4,
             'weight_decay':1e-3}

logvar_params = {'params':net.conv_logvar.parameters(),
                 'lr':1e-4,
                 'weight_decay':1e-3}

decoder_params = {'params':net.decoder.parameters(),
                 'lr':1e-4,
                 'weight_decay':1e-3}

parameters = [preprocessing_params, encoder_params, mu_params,
              logvar_params, decoder_params]

optimizer = torch.optim.Adam(params = parameters)

# initialize learning rate scheduler
    
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
#                                           step_size = 3,
#                                           gamma = 0.5,
#                                           last_epoch = -1)

# whether to sample a single batch for a trial run

trial_run = False

# otherwise, set the number of epochs to train and validate for

if not trial_run:
    num_epochs = 1

# record the best loss across epochs

best_val_loss = 1e10
best_mse_loss = 0
best_kld_loss = 0

# starting time

start = time.time()

beta = 0.01

if __name__ == '__main__':

    if trial_run:
        
        # record the epoch start time
        
        epoch_start = time.time()
        
        # training ###########################################################
        
        print('\nTraining...')
        
        # put net in training mode
        
        net.train()
        
        # sample a batch
        
        loader = dataloaders['train']
        x = next(iter(loader))
        
        # move to GPU
            
        x = x.to(device)
        
        # train over the batch
        
        MSE,KLD = train_batch(x,beta,device,net,loss_func,optimizer)
        
        # show results
        
        print('MSE: {:.5f}'.format(MSE))
        print('KLD: {:.5f}'.format(KLD))
        
        # validation #########################################################
        
        print('\nValidating...')
        
        # put net in testing mode
                      
        net.eval()
        
        # sample a batch
        
        loader = dataloaders['val']
        x = next(iter(loader))
        
        # move to GPU
        
        x = x.to(device)
        
        # validate over the batch
        
        MSE,KLD = val_batch(x,beta,device,net,loss_func)
        
        # show results
        
        print('MSE: {:.5f}'.format(MSE))
        print('KLD: {:.5f}'.format(KLD))
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        
        # to compare to best validation loss
        
        val_loss = MSE - KLD
        
        # save the weights for the best validation loss
        
        if val_loss < best_val_loss:
            
            print('Saving checkpoint...')
            
            best_val_loss = val_loss
            
            # deepcopy needed because a dict is a mutable object
            
            best_parameters = copy.deepcopy(net.state_dict())
            
            torch.save(best_parameters,'best_param.pt')
        
        # testing ############################################################
        
        print('\nTesting...')
        
        # load best parameters
    
        net.load_state_dict(torch.load('best_param.pt'))
        
        # put net in testing mode
        
        net.eval()
        
        # sample a batch
        
        loader = dataloaders['test']
        x = next(iter(loader))
        
        # move to GPU
        
        x = x.to(device)
        
        # test the batch
        
        MSE,KLD = val_batch(x,beta,device,net,loss_func)
        
        # show results
        
        print('MSE: {:.5f}'.format(MSE))
        print('KLD: {:.5f}'.format(KLD))
        
    else:
        
        for epoch in range(num_epochs):
            
            # show number of epochs elapsed
            
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            
            # record the epoch start time
            
            epoch_start = time.time()
            
            # train for an epoch
        
            mse_loss,kld_loss = train_epoch(net,beta,dataloaders['train'],
                                            device,loss_func,optimizer)
            
            # show results
            
            print('\nMSE: {:.5f}'.format(mse_loss))
            print('KLD: {:.5f}'.format(kld_loss))
            
            # validate for an epoch
            
            mse_loss,kld_loss = val_epoch(net,beta,dataloaders['val'],device,
                                          loss_func)
            
            # show results
            
            print('\nMSE: {:.5f}'.format(mse_loss))
            print('KLD: {:.5f}'.format(kld_loss))
            
            # update learning rate
            
            # scheduler.step()
            
            # show epoch time
            
            epoch_end = time.time()
            
            epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
            
            print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
            
            # to compare to best validation loss
            
            val_loss = mse_loss - kld_loss
            
            # save the weights for the best validation loss
        
            if val_loss < best_val_loss:
                
                print('Saving checkpoint...')
                
                best_val_loss = val_loss
                best_mse_loss = mse_loss
                best_kld_loss = kld_loss
                
                # deepcopy needed because a dict is a mutable object
                
                best_parameters = copy.deepcopy(net.state_dict())
                
                torch.save(best_parameters,'best_param.pt')
             
            if beta < 1.0:
                beta *= 2
        
        # show training and validation time and best validation accuracy
        
        end = time.time()
        total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
        print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
        print('Best MSE Loss: {:.5f}'.format(best_mse_loss))
        print('Best KLD Loss: {:.5f}'.format(best_kld_loss))
