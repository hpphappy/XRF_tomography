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

params_3d_64_64_64_nElements_1 = {'n_ranks': n_ranks,
                                    'rank': rank,
                                    'P_folder':'../data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',
                                    'f_P':'Intersecting_Length_64_64_64', 
                                    'theta_st': tc.tensor(0).to(dev), 
                                    'theta_end': tc.tensor(2 * np.pi).to(dev),
                                    'n_theta': tc.tensor(200).to(dev),
                                    'src_path': os.path.join('../data/sample_9_size_64_pad/nElements_1', 'grid_concentration.npy'),
                                    'det_size_cm': 0.9,
                                    'det_from_sample_cm': 1.6,
                                    'det_ds_spacing_cm': 0.4,
                                    'sample_size_n': tc.tensor(64).to(dev),
                                    'sample_size_cm': tc.tensor(0.01).to(dev),                   
                                    'sample_height_n': tc.tensor(64).to(dev),
                                    'this_aN_dic': {"Si": 14},
                                    'probe_cts': tc.tensor(1.0E7).to(dev),
                                    'probe_energy': np.array([20.0]),                    
                                    'save_path': '../data/sample_9_size_64_data/nElements_1',
                                    'save_fname': 'XRF_sample9',
                                    'Poisson_noise': True,                
                                    'dev': dev
                                   }

params_3d_44_44_20_xtal1_roi_plus = {'n_ranks': n_ranks,
                                    'rank': rank,
                                    'P_folder':'../data/P_array/sample_44_44_20_n/Dis_3.0_detSize_2.4_manual_dpts_4',
                                    'f_P':'Intersecting_Length_44_44_20', 
                                    'theta_st': tc.tensor(165 * np.pi/180).to(dev), 
                                    'theta_end': tc.tensor(-165 * np.pi/180).to(dev),
                                    'n_theta': tc.tensor(110).to(dev),
                                    'src_path': os.path.join('../data/Xtal1_align1_adjusted1_ds4_recon/Ab_T_nEl_4_nDpts_4_b_1E-1_lr_1E-4_ini_1_cont_1', 'grid_concentration.npy'),
                                    'det_size_cm': 2.4,
                                    'det_from_sample_cm': 3.0,
                                    'det_ds_spacing_cm': 1.2,
                                    'sample_size_n': tc.tensor(44).to(dev),
                                    'sample_size_cm': tc.tensor(0.007).to(dev),                   
                                    'sample_height_n': tc.tensor(20).to(dev),
                                    'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                                    'probe_cts': tc.tensor(2.3E5).to(dev),
                                    'probe_energy': np.array([10.0]),                    
                                    'save_path': '../data/Xtal1_align1_adjusted1_ds4_recon/Ab_T_nEl_4_nDpts_4_b_1E-1_lr_1E-4_ini_1_cont_1/reprojected',
                                    'save_fname': 'XRF_xtal1_ds4',
                                    'Poisson_noise': False,                
                                    'dev': dev
                                   }


params = params_3d_64_64_64_nElements_1


if __name__ == "__main__":  
    XRF_data  = create_XRF_data_3d(**params)
    
    if rank == 0:
        save_path = params["save_path"]
        with open(os.path.join(save_path, 'XRF_data_parameters.txt'), "w") as recon_paras:
            print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')