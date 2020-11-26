import torch
import numpy as np
import pandas as pd

class CSVDataset(torch.utils.data.Dataset):
    
    def __init__(self,x_csv_path,window_length=160,window_overlap=0.5):
        
        # load csv file as dataframe
        
        self.x = pd.read_csv(x_csv_path,header = None)
        
        # specify how many samples in a window
        
        self.window_length = window_length
        
        # specify overlap between windows as a fraction
        
        self.window_overlap = window_overlap
    
    def __len__(self):
        
        self.last = int(((len(self.x) - self.window_length) / 
                     (self.window_length*(1-self.window_overlap))) + 1)
        
        return self.last
        
    def __getitem__(self,idx):
        
        # get sample numbers for window start and end        
        
        start = int(idx*self.window_length*(1-self.window_overlap))
        end = int((idx*self.window_length*(1-self.window_overlap)) + 
                  self.window_length)
        
        # extract window and convert to NumPy array
        
        window = self.x.iloc[start:end].to_numpy(dtype = np.float64,
                                                 copy = True)
        
        # convert window to torch tensor
        
        window = torch.from_numpy(window).unsqueeze(dim=0)
        
        return window
            
if __name__ == '__main__':
    
    path = '../../../data/projF/train/x_train.csv'
    dataset = CSVDataset(path)
    
    x = dataset[1]
