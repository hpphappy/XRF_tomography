 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch as tc
import xraylib as xlib
from XRF_tomography import reconstruct_jXRFT_tomography
from mpi4py import MPI
from misc import create_summary

import warnings

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()
warnings.filterwarnings("ignore")

#========================================================
# Set the device
#========================================================
# stdout_options = {'output_folder': recon_path, 'save_stdout': False, 'print_terminal': True}
gpu_index = rank % 2
# gpu_index = 1
if tc.cuda.is_available():  
    dev = tc.device('cuda:{}'.format(gpu_index))
    print("Process ", rank, "running on", dev)
    sys.stdout.flush()
else:  
    dev = "cpu"
    print("Process", rank, "running on CPU")
    sys.stdout.flush()


fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
      "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
      "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
     }


params_3d_test_sample8_64_64_64 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': False,
                              'probe_intensity': 1.0E7,
                              'std_path': None,
                              'f_std': None,
                              'std_element_lines_roi': None,
                              'density_std_elements': None,
                              'fitting_method': None,
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/sample_8_size_64_test_recon_Ab_XRFT_b1_10E4_b2_1E0',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/sample_8_size_64_test',
                              'f_XRF_data': 'test8_xrf',                  
                              'f_XRT_data': 'test8_xrt',
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Ca": 20, "Sc": 21},
                              'element_lines_roi': np.array([['Ca', 'K'], ['Ca', 'L'], ['Sc', 'K'], ['Sc', 'L']]),
                              'n_line_group_each_element': np.array([2, 2]),
                              'sample_size_n': 64, 
                              'sample_height_n': 64,
                              'sample_size_cm': 0.01,                                    
                              'probe_energy': np.array([20.0]),                            
                              'n_epochs': 300,
                              'save_every_n_epochs': 1,
                              'minibatch_size': 64,
                              'b1': 10000,  # the regulizer coefficient of the XRT loss
                              'b2': 1,
                              'lr': 1.0E-3,                          
                              'det_dia_cm': 0.9,
                              'det_from_sample_cm': 1.6,
                              'manual_det_coord': False,
                              'set_det_coord_cm': None,
                              'det_on_which_side': "positive", 
                              'manual_det_area': False,
                              'det_area_cm2': None, 
                              'det_ds_spacing_cm': 0.4,
                              'P_folder': 'data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                              'f_P': 'Intersecting_Length_64_64_64',
                              'fl_K': fl["K"],
                              'fl_L': fl["L"],                
                              'fl_M': fl["M"]
                             }


params_3d_test_sample9_64_64_64 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/sample_9_size_64_recon/b_1.5E-5_lr_1.0E-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/sample_9_size_64_data/nElements_1',
                              'f_XRF_data': 'test9_xrf',               
                              'f_XRT_data': 'test9_xrt',
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,
                              'theta_ls_dataset': 'exchange/theta',
                              'channel_names': 'exchange/elements',
                              'this_aN_dic': {"Si": 14},
                              'element_lines_roi': np.array([['Si', 'K'], ['Si', 'L']]),
                              'n_line_group_each_element': np.array([2]),
                              'sample_size_n': 64, 
                              'sample_height_n': 64,
                              'sample_size_cm': 0.01,                                    
                              'probe_energy': np.array([20.0]),                            
                              'n_epochs': 40,
                              'minibatch_size': 64,
                              'b1': 1.5E-5,
                              'b2': 1.0,
                              'lr': 1.0E-3,                          
                              'det_dia_cm': 0.9,
                              'det_from_sample_cm': 1.6,
                              'manual_det_coord': False,
                              'set_det_coord_cm': None,
                              'det_on_which_side': "positive", 
                              'manual_det_area': False,
                              'det_area_cm2': None, 
                              'det_ds_spacing_cm': 0.4,
                              'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                              'f_P': 'Intersecting_Length_64_64_64',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }


params = params_3d_test_sample8_64_64_64

if __name__ == "__main__": 
    
    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        create_summary(output_folder, params)
