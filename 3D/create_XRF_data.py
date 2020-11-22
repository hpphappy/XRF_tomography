#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:58:22 2020

@author: panpanhuang
"""

import os
import numpy as np
import torch as tc
from data_generation_fns import create_XRF_data_3d
import warnings
warnings.filterwarnings("ignore")

dev = "cpu"



params_3d_5_5_5 = {'P_save_path': os.path.join('./data/sample3_recon', 'Intersecting_Length_5_5_5'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(16).to(dev),
                   'src_path': os.path.join('./data/sample3_pad', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(5).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(5).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample3_data',
                   'save_fname': 'XRF_sample3',
                   'dev': dev
                  }

params_3d_64_64_64 = {'P_save_path': os.path.join('./data/sample1_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample1_pad', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample1_data',
                   'save_fname': 'XRF_sample1',
                   'dev': dev
                  }


params = params_3d_5_5_5


if __name__ == "__main__":  
    XRF_data  = create_XRF_data_3d(**params)

    save_path = params["save_path"]
    with open(os.path.join(save_path, 'XRF_data_parameters.txt'), "w") as recon_paras:
        print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')