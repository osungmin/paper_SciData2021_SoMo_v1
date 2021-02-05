import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class load_forcing(Dataset):

    def __init__(self, ismn_root, pixel, dates, n_inputs, seq_length,
                 attribute_means, attribute_stds):

        self.ismn_root = ismn_root
        self.pixel = pixel
        self.n_inputs = n_inputs
        self.seq_length = seq_length
        self.dates = dates
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        
        self.x = self.load_forcing()
        self.attributes = self.load_static()

        self.n_samples = self.x.shape[0]


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.attributes


    def load_forcing(self):

        file_path = "./ismn/forcing/features_"+(self.pixel.split('IDX_')[1]).split('_dep')[0]+".dat"
        if not os.path.isfile(file_path):
           raise RuntimeError(f'No file for Pixel {self.pixel} at {file_path}')

        # load forcing data
        df = pd.read_csv(file_path, 
                          sep=',', index_col=0, header=0, 
                          parse_dates=True, na_values=-9999.)

        # select forcing data period
        start_date = self.dates[0] - pd.DateOffset(days=self.seq_length - 1)
        end_date = self.dates[1]

        df = df[start_date:end_date]

        # use all meteorological variables as inputs
        if  self.n_inputs==3: selected_cols= ["log_tp","t2m","snr"]
        elif  self.n_inputs==6: selected_cols= ["log_tp","t2m","snr","ssrd","q","skt"]
        else: raise RuntimeError(f'Check number of forcing data, n_inputs {n_inputs}')

        
        # input selected
        _x = np.array([df[col].values for col in selected_cols]).T

        # normalise forcing data
        df1= pd.read_csv('./ismn/lis/scaler_ALL_layer0.dat', sep=',', index_col=0, header=0)


        SCALER = {'means': df1.loc["aveK",selected_cols].values,
                  'stds': df1.loc["stdK",selected_cols].values}

        _x = (_x - SCALER["means"]) / SCALER["stds"]

        n_samples, n_features = _x.shape

        x = np.zeros((n_samples - self.seq_length + 1, self.seq_length, n_features))

        # reshape
        for i in range(0, x.shape[0]):
           x[i, :, :n_features] = _x[i:i + self.seq_length, :]


        return torch.from_numpy(x.astype(np.float32))


    def load_static(self):

        df = pd.read_csv('./ismn/attri/global_attri.dat', sep=',', index_col=0, header=0)

        # normalize data
        df = (df - self.attribute_means) / self.attribute_stds

        # store feature as PyTorch Tensor
        pixel=((self.pixel).split('IDX_')[1]).split('_dep')[0]
        attributes = df.loc[df.index == pixel].values 

        return torch.from_numpy(attributes.astype(np.float32))



def rescale_features(feature):

    df= pd.read_csv('./ismn/lis/scaler_ALL_layer0.dat', sep=',', index_col=0, header=0)

    feature = feature * df.loc['stdK','sm'] + df.loc['aveK','sm']
    return feature

