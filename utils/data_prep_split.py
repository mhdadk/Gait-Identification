import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import minmax_scale
import numpy as np
import os

x_csv = '../../../data/projF/train/x.csv'
y_csv = '../../../data/projF/train/y.csv'

x = pd.read_csv(x_csv,header = None)
y = pd.read_csv(y_csv,header = None).to_numpy()

# upsample then truncate y to match number of samples in x

y = np.repeat(y,4,axis=0)[:-6]
y = pd.DataFrame(data = y)

# x = x.to_numpy()
# x = minmax_scale(x,axis=1)
# x = pd.DataFrame(data = x)

x_train,x_val,y_train,y_val = train_test_split(x,y,
                                               train_size = 0.8,
                                               random_state = 42,
                                               shuffle = False)

x_val,x_test,y_val,y_test = train_test_split(x_val,y_val,
                                             train_size = 0.5,
                                             random_state = 42,
                                             shuffle = False)

# np.savetxt('x_train.csv',x,delimiter=',')

if not os.path.exists('../../../data/projF/train/x_train.csv'):
    
    x_train.to_csv('../../../data/projF/train/x_train.csv',
                    header = False,
                    index = False)

if not os.path.exists('../../../data/projF/train/y_train.csv'):
    
    y_train.to_csv('../../../data/projF/train/y_train.csv',
                    header = False,
                    index = False)

if not os.path.exists('../../../data/projF/train/x_val.csv'):
    
    x_val.to_csv('../../../data/projF/train/x_val.csv',
                 header = False,
                 index = False)

if not os.path.exists('../../../data/projF/train/y_val.csv'):
    
    y_val.to_csv('../../../data/projF/train/y_val.csv',
                 header = False,
                 index = False)

if not os.path.exists('../../../data/projF/train/x_test.csv'):
    
    x_test.to_csv('../../../data/projF/train/x_test.csv',
                  header = False,
                  index = False)

if not os.path.exists('../../../data/projF/train/y_test.csv'):
    
    y_test.to_csv('../../../data/projF/train/y_test.csv',
                  header = False,
                  index = False)