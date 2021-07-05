#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:49:56 2020

@author: panpanhuang
"""

import os
import numpy as np
import torch as tc
from data_generation_fns_mpi_updating_h5Parray import create_XRT_data_3d
import warnings
warnings.filterwarnings("ignore")

dev = "cpu"




params_3d_44_44_20_xtal1_roi_plus = { 'src_path': os.path.join('../data/Xtal1_align1_adjusted1_ds4_recon/Ab_F_nEl_1_nDpts_4_b_0.0_lr_1E-3_set_ini_1', 'initialized_grid_concentration.npy'),
                    'theta_st': tc.tensor(165 * np.pi/180).to(dev), 
                    'theta_end': tc.tensor(-165 * np.pi/180).to(dev),
                    'n_theta': tc.tensor(110).to(dev),
                   'sample_height_n': tc.tensor(20).to(dev),
                   'sample_size_n': tc.tensor(44).to(dev),
                   'sample_size_cm': tc.tensor(0.007).to(dev),
                   'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                   'probe_energy': np.array([10.0]),
                   'probe_cts': tc.tensor(2.3E5).to(dev), 
                   'save_path': '../data/Xtal1_align1_adjusted1_ds4_recon/Ab_F_nEl_1_nDpts_4_b_0.0_lr_1E-3_set_ini_1/reprojected',
                   'save_fname': 'XRT_xtal1_ds4',
                   'theta_sep': False,
                   'Poisson_noise': False,             
                   'dev': dev
                  }



params_3d_32_32_32_xtal1_fake = { 'src_path': os.path.join('../data/Xtal1_fake_sample', 'grid_concentration.npy'),
                    'theta_st': tc.tensor(165 * np.pi/180).to(dev), 
                    'theta_end': tc.tensor(-165 * np.pi/180).to(dev),
                    'n_theta': tc.tensor(110).to(dev),
                   'sample_height_n': tc.tensor(32).to(dev),
                   'sample_size_n': tc.tensor(32).to(dev),
                   'sample_size_cm': tc.tensor(0.007).to(dev),
                   'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                   'probe_energy': np.array([10.0]),
                   'probe_cts': tc.tensor(2.3E5).to(dev), 
                   'save_path': '../data/Xtal1_fake_sample',
                   'save_fname': 'XRT_xtal1_fake_sample',
                   'theta_sep': True,
                   'Poisson_noise': True,             
                   'dev': dev
                  }

    
params = params_3d_32_32_32_xtal1_fake


if __name__ == "__main__":  
    XRT_data  = create_XRT_data_3d(**params)

    save_path = params["save_path"]
    with open(os.path.join(save_path, 'XRT_data_parameters.txt'), "w") as recon_paras:
        print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')
