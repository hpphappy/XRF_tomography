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



params_124_124_32_cabead = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Cabead/axo_std',
                              'f_std': 'axo_std.mda.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': True,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Cabead_adjusted1_ds4_recon/Ab_F_nEl_6_nDpts_4_b1_1e2_b2_1e0_lr_1.0e-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Cabead_adjusted1_ds4',
                              'f_XRF_data': 'cabead_xrf-fits',                 
                              'f_XRT_data': 'cabead_scalers',
                              'scaler_counts_us_ic_dataset_idx':18,
                              'scaler_counts_ds_ic_dataset_idx':11,
                              'XRT_ratio_dataset_idx':21,   
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',  
                              'this_aN_dic': {"Si": 14, "Ti": 22, "Cr": 24, "Fe": 26, "Ni":28, "Ba": 56},
                              'element_lines_roi': np.array([['Si', 'K'], ['Ti', 'K'], ['Cr', 'K'],
                                                             ['Fe', 'K'], ['Ni', 'K'], ['Ba', 'L']]),  # np.array([["Si, K"], ["Ca, K"]])
                              'n_line_group_each_element': np.array([1, 1, 1, 1, 1, 1]),
                              'sample_size_n': 124, 
                              'sample_height_n': 32,
                              'sample_size_cm': 0.0248,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 100,
                              'save_every_n_epochs': 5,
                              'minibatch_size': 124,
                              'b1': 1.0E2, 
                              'b2': 1.0E0,
                              'lr': 1.0E-3,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, 1.69, 0.70], [0.70, 1.69, -0.70], [-0.70, 1.69, 0.70], [-0.70, 1.69, -0.70]]),
                              'det_on_which_side': "positive",
                              'det_from_sample_cm': None,
                              'det_ds_spacing_cm': None,
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,
                              'det_dia_cm': None,
                              'P_folder': 'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_124_124_32',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }


params_124_124_32_cabead_2 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Cabead/axo_std',
                              'f_std': 'axo_std.mda.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': False,
                              'cont_from_check_point': True,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Cabead_adjusted1_ds3_recon/Ab_F_ProbeAtt_F_nEl_6_nDpts_3_b1_0.0_lr_1.0e-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Cabead_adjusted1_ds4',
                              'f_XRF_data': 'cabead_xrf-fits',                 
                              'f_XRT_data': 'cabead_scalers',
                              'scaler_counts_us_ic_dataset_idx':18,
                              'scaler_counts_ds_ic_dataset_idx':11,
                              'XRT_ratio_dataset_idx':21,   
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',  
                              'this_aN_dic': {"Si": 14, "Ti": 22, "Cr": 24, "Fe": 26, "Ni":28, "Ba": 56},
                              'element_lines_roi': np.array([['Si', 'K'], ['Ti', 'K'], ['Cr', 'K'],
                                                             ['Fe', 'K'], ['Ni', 'K'], ['Ba', 'L']]),  # np.array([["Si, K"], ["Ca, K"]])
                              'n_line_group_each_element': np.array([1, 1, 1, 1, 1, 1]),
                              'sample_size_n': 124, 
                              'sample_height_n': 32,
                              'sample_size_cm': 0.0248,                                    
                              'probe_energy': np.array([10.0]),
                              'probe_att': False,
                              'n_epochs': 100,
                              'save_every_n_epochs': 5,
                              'minibatch_size': 124,
                              'b1': 0.0, 
                              'b2': 0.0,
                              'lr': 1.0E-3,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, 1.69, 0.70], [-0.70, 1.69, 0.70], [-0.70, 1.69, -0.70]]),
                              'det_on_which_side': "positive",
                              'det_from_sample_cm': None,
                              'det_ds_spacing_cm': None,
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,
                              'det_dia_cm': None,
                              'P_folder': 'data/P_array/sample_124_124_32/Dis_1.69_manual_dpts_3',              
                              'f_P': 'Intersecting_Length_124_124_32',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params = params_124_124_32_cabead_2

if __name__ == "__main__": 
    
    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        create_summary(output_folder, params)
