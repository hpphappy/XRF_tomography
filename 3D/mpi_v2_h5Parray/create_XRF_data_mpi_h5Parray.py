#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:58:22 2020

@author: panpanhuang
"""

import os
import numpy as np
from mpi4py import MPI
import torch as tc
from data_generation_fns_mpi_updating_h5Parray import create_XRF_data_3d
import warnings
warnings.filterwarnings("ignore")

dev = "cpu"
comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

params_3d_64_64_64_nElements_2_2 = {'n_ranks': n_ranks,
                                    'rank': rank,
                                    'P_folder':'./data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5/backup4',
                                    'f_P':'Intersecting_Length_64_64_64', 
                                    'theta_st': tc.tensor(0).to(dev), 
                                    'theta_end': tc.tensor(2 * np.pi).to(dev),
                                    'n_theta': tc.tensor(200).to(dev),
                                    'src_path': os.path.join('./data/sample8_size_64_pad/nElements_2', 'grid_concentration.npy'),
                                    'det_size_cm': 0.9,
                                    'det_from_sample_cm': 1.6,
                                    'det_ds_spacing_cm': 0.4,
                                    'sample_size_n': tc.tensor(64).to(dev),
                                    'sample_size_cm': tc.tensor(0.01).to(dev),                   
                                    'sample_height_n': tc.tensor(64).to(dev),
                                    'this_aN_dic': {"Ca": 20, "Sc": 21},
                                    'probe_cts': tc.tensor(1.0E7).to(dev),
                                    'probe_energy': np.array([20.0]),                    
                                    'save_path': './data/sample8_size_64_data/nElements_2/nThetas_200_limitedSolidAngle/solidAngle_frac_0.0156/woNoise',
                                    'save_fname': 'XRF_sample8',
                                    'Poisson_noise': False,                
                                    'dev': dev
                                   }


params = params_3d_64_64_64_nElements_2_2


if __name__ == "__main__":  
    XRF_data  = create_XRF_data_3d(**params)
    
    if rank == 0:
        save_path = params["save_path"]
        with open(os.path.join(save_path, 'XRF_data_parameters.txt'), "w") as recon_paras:
            print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')