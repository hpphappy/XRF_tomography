#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch as tc
import xraylib as xlib
from XRF_tomography_mpi import reconstruct_jXRFT_tomography

import warnings
warnings.filterwarnings("ignore")


#========================================================
# Set the device
#========================================================
# if tc.cuda.is_available():  
#     dev = "cuda:1" 
#     print("running on GPU")
# else:  
#     dev = "cpu"
#     print("running on CPU")

# device = tc.device(dev)
# print("device = %s" %device)

dev = "cpu"


fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE])

fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE])

fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

fl_line_groups = np.array(["K", "L", "M"])
group_lines = True



params_3d_5_5_5 = {'dev': dev,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample3_recon',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample3_pad',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample3_data',
                      'f_XRF_data': 'XRF_sample3',                     
                      'f_XRT_data': 'XRT_sample3',
                      'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(40).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev) ,
                      'minibatch_size': tc.tensor(5).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(16).to(dev),
                      'sample_size_n': tc.tensor(5).to(dev), 
                      'sample_height_n': tc.tensor(5).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.24,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.1,
                      'f_P': 'Intersecting_Length_5_5_5',
                     }


params_3d_64_64_64 = {'dev': dev,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample1_recon',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample1_pad',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample1_data',
                      'f_XRF_data': 'XRF_sample1',                     
                      'f_XRT_data': 'XRT_sample1',
                      'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(40).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(256).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(200).to(dev),
                      'sample_size_n': tc.tensor(64).to(dev), 
                      'sample_height_n': tc.tensor(64).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.24,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.1,
                      'f_P': 'Intersecting_Length_64_64_64',
                     }


params = params_3d_64_64_64


if __name__ == "__main__": 
    
    reconstruct_jXRFT_tomography(**params)





