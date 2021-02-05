#!/usr/bin/env python
"""
LSTM-based soil moisture simulation model used in

O, S. and Orth, R., Global soil moisture data derived through machine learning trained with in-situ measurements, 
Sci. Data (2021)

Please note that the LSTM model (e.g., ) is obtained from


"""
import os
#os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID" #to use GPU
#os.environ['CUDA_VISIBLE_DEVICES']="0" #select GPU#0

import pickle
import sys

from datetime import datetime
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.subs import (load_forcing, rescale_features)
from codes.ealstm import EALSTM


def set_cfg():
    # set model run parameters
    cfg={ 'ismn_root': './ismn/',
          'run_dir' : 'model/',
          'n_inputs' : 6,
          'seq_length' : 365,
          'run_epoch' : 15,
          'hidden_size': 128,
          'dropout': 0.5,
          'run_start': pd.to_datetime('01012000', format='%d%m%Y'),
          'run_end':  pd.to_datetime('31122019', format='%d%m%Y')
        }

    # convert path to PosixPath object
    cfg["ismn_root"] = Path(cfg["ismn_root"])
    cfg["run_dir"] = Path(__file__).absolute().parent / cfg["run_dir"]

    return cfg


def load_pixel_lis():

    # part of global pixel
    pixel_file = Path(__file__).absolute().parent / "ismn/lis/run_global_grid0.lis"

    with pixel_file.open('r') as fp:
        pixels = fp.readlines()
    pixels = [pixel.strip() for pixel in pixels]

    return pixels


"""
def load_static(db_path: str,
                pixels: []):

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'pixel_attributes'", conn)

    pixels=[(x.split('IDX_')[1]).split('_dep')[0] for x in pixels]

    df = df[df.index.isin(pixels)]
    df = df.drop(['index'], axis=1)

    return df
"""


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout

        self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                           input_size_stat=input_size_stat,
                           hidden_size=hidden_size,
                           initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc  = nn.Linear(hidden_size, 1)


    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None):

        h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])

        out = self.fc(last_h)

        return out, h_n, c_n


def run(run_cfg):
    """Run model

    ----------
    user_cfg : Dict
        Model run config

    """

    print("\n\n***** loaded run_cfg")
    print(run_cfg)

    ### load grid pixel list 
    pixels = load_pixel_lis()
    print("\n >>> len of pixels to run the model over:", len(pixels))    

    ### scaled with means and stds of train attributes
    print(" >>> load static values for scaler")
    att_path   = str(run_cfg["ismn_root"] / "attri/scaler_ALL_statics.dat")
    attri_scaler = pd.read_csv(att_path, sep=',', index_col=0, header=0)

    means = attri_scaler.iloc[0] 
    stds  = attri_scaler.iloc[1]
   

    # create model
    model = Model(input_size_dyn=6,
                  input_size_stat=10,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"]).to(DEVICE)


    # load trained model
    weight_file = run_cfg["run_dir"] / 'model_epoch15.pt'
    print(" >>> selected model:", weight_file)

    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
     
    results = {}


    for pixel in tqdm(pixels):
        ds_test = load_forcing(ismn_root=run_cfg["ismn_root"],
                               pixel=pixel,
                               dates=[run_cfg["run_start"],run_cfg["run_end"]],
                               n_inputs=run_cfg["n_inputs"],
                               seq_length=run_cfg["seq_length"],
                               attribute_means=means,
                               attribute_stds=stds)


        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=3)

        preds = run_over_pixel(run_cfg, model, loader)

        df = pd.DataFrame(data={'somo': preds.flatten()})#, index=date_range)

        results[pixel] = df



    file_name = run_cfg["run_dir"] / f"somo_out.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Results stored at {file_name}")


def run_over_pixel(run_cfg, model, loader):

    model.eval()

    preds= None

    with torch.no_grad():
        for data in loader:

            x_d, x_s = data
            x_d, x_s = x_d.to(DEVICE), x_s.to(DEVICE)
            p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)

        # rescale output using scaler
        preds = rescale_features(preds.numpy())

        # soil moisture <9 to zero
        preds[preds < 0] = 0

    return preds



if __name__ == "__main__":


    DEVICE = torch.device('cpu') #in case, gpu is available, e.g., 'cuda:0'

    cfg = set_cfg()

    # check parameters 
    print("\n\n ****** Running the SoMo.ml model ***** \n\n")
    print(" > device:", DEVICE, "\n")
    print(cfg)

    print()
    wait=input()

    globals()['run'](cfg)













