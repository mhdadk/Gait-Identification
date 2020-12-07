import data_manipulation.data_loading as dl
import data_manipulation.preprocessing as prep
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import os
from generative1D.variational_autoencoder import *

training_folder = '../data/TrainingData'
#Parameters
offset = 64
validation_split = 0.1
base_augment = 1
basepath = '../data/AugmentedTraining'
model_parameters = {
    'latent_space_size' : 32,
    #'encoder': CNNLSTMEncoder,
    'weights_file': '../results/001-baseline-base-low_kl-z32/best_params.pt',
    #'decoder': Decoder2
}
def balanced_resample(data, labels, resample, divide_size=1):
    """! Resample based on class"""
    x = []
    y = []
    for i, r in enumerate(resample):
        for _ in range(r):
            class_subset = data[labels == i]
            x.append(class_subset)
            y.append(labels[labels == i])
    x = np.concatenate(x)
    y = np.concatenate(y)
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    x = x[:len(y)//divide_size] # cut dataset in half 
    y = y[:len(y)//divide_size]
    return x,y

def main():
    """! Generate data according to set parameters"""
    # load training set
    training_files = dl.get_and_sort_files(training_folder)
    _, xdata, _, ydata = dl.load_and_segment_data(training_files, xoffset=offset)# time isnt important yet
    # calculate resampling
    resample = [base_augment]
    for i in range(1,4):
        resample.append(base_augment * int(np.ceil(np.sum(ydata==0)/np.sum(ydata==i))))
    # Separate train and validation in a deterministic way
    indices = list(range(xdata.shape[0]))
    split = int(np.floor(0.1 * xdata.shape[0])) 
    train_indices, val_indices = indices[split:], indices[:split]
    xtrain = xdata[train_indices]
    xval = xdata[val_indices]
    ytrain = ydata[train_indices]
    yval = ydata[val_indices]
    
    # Load Model
    gen = BasicVAE(**model_parameters)

    # Construct preprocessing steps
    prep_steps = [    
        prep.SetZeroMean(),
        prep.NormalizeMinMax(),
    ]   
    for step in prep_steps:
        xtrain, ytrain = step.process(xtrain, ytrain)
        xval, yval = step.process(xval, yval)
    #shuffle data
    p = np.random.permutation(len(ytrain))
    xtrain = xtrain[p]
    ytrain = ytrain[p]
    p = np.random.permutation(len(yval))
    xval = xval[p]
    yval = yval[p]
    #resample
    xtrain, ytrain = balanced_resample(xtrain, ytrain, resample)
    xval, yval = balanced_resample(xval, yval, resample)
    # convert to tensors
    xtrain = torch.from_numpy(xtrain).double()
    xval = torch.from_numpy(xval).double()
    xtrain = gen.get_latent(xtrain).detach().numpy()
    xval = gen.get_latent(xval).detach().numpy()
    #Assert normalization
    #assert np.min(xtrain) >=0
    #assert np.max(xtrain) <=1
    #Assert Size Parity Preservation
    assert len(xtrain) == len(ytrain)
    assert len(xval) == len(yval)
    # Save Data 
    filepath = os.path.join(basepath, 'baseline_model_dec1_LReg0-70_z32_low_kl_schedule2.h5')
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('train_data', data=xtrain)
        f.create_dataset('train_labels', data=ytrain)
        f.create_dataset('val_data', data=xval)
        f.create_dataset('val_labels', data=yval)
        
if __name__ == "__main__":
    main()