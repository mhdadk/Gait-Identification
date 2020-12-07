import torch
import numpy as np
import pandas as pd

class CSVDataset(torch.utils.data.Dataset):
    
    def __init__(self,x_csv_path, window_length = 160,
                 window_overlap = 0.5):
        
        # load data as dataframe
        
        self.x = pd.read_csv(x_csv_path,header = None)
        
        # specify how many samples in a window
        
        self.window_length = window_length
        
        # specify overlap between windows as a fraction
        
        self.window_overlap = window_overlap
    
    def __len__(self):
        
        # note that this drops the last window with length less than
        # self.window_length
        
        last = int(((len(self.x) - self.window_length) / 
                     (self.window_length*(1-self.window_overlap))) + 1)
        
        return last
        
    def __getitem__(self,idx):
        
        # get sample numbers for window start and end        
        
        start = int(idx*self.window_length*(1-self.window_overlap))
        end = int((idx*self.window_length*(1-self.window_overlap)) + 
                  self.window_length)
        
        # extract window and convert to NumPy array
        
        x_window = self.x.iloc[start:end].to_numpy(dtype = np.float64,
                                                   copy = True)
        
        # convert window to torch tensor
        
        x_window = torch.from_numpy(x_window)
        
        # zero mean each channel to avoid outliers in spectrograms and
        # transpose data to be a num_channels x window_length tensor
        
        x_window = x_window.transpose(0,1) - torch.mean(x_window,dim=1)
        
        return x_window

if __name__ == '__main__':
    
    x_path = '../../../data/projF/train/x_train.csv'
    dataset = CSVDataset(x_path)
    
    x = dataset[len(dataset)-1]
