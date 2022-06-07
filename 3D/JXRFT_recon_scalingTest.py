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





params_3d_size_32 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': False,
                              'probe_intensity': 1.0E7,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': None,
                              'density_std_elements': None,
                              'fitting_method': None,                         
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/size_32_recon/Ab_T_nEl_2_Dis_1.6_nDpts_5_b1_1.5E-5_b2_1.0_lr_1.0E-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/size_32/n_element_2',
                              'f_XRF_data': 'simulation_XRF_data.h5',                  
                              'f_XRT_data': 'simulation_XRT_data.h5', 
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',
                              'this_aN_dic': {"Ca": 20, "Sc": 21},
                              'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                              'n_line_group_each_element': np.array([1,1]),
                              'sample_size_n': 32, 
                              'sample_height_n': 32,
                              'sample_size_cm': 0.01,                                    
                              'probe_energy': np.array([20.0]),                            
                              'n_epochs': 2,
                              'save_every_n_epochs': 1,
                              'minibatch_size': 32,
                              'b1': 1.5E-5,
                              'b2': 1.0,
                              'lr': 1.0E-3,        
                              'manual_det_coord': False,    
                              'set_det_coord_cm': None,            
                              'det_on_which_side': "positive",   
                              'det_from_sample_cm': 1.6,
                              'det_ds_spacing_cm': 0.4,
                              'manual_det_area': False,
                              'det_area_cm2': None, 
                              'det_dia_cm': 0.9,                        
                              'P_folder': './data/P_array/sample_32_32_32/detSpacing_0.4_dpts_5',              
                              'f_P': 'Intersecting_Length_32_32_32',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_3d_size_64 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': False,
                              'probe_intensity': 1.0E7,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': None,
                              'density_std_elements': None, 
                              'fitting_method': None,                         
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/size_64_recon/Ab_T_nEl_2_Dis_1.6_nDpts_5_b1_1.5E-5_b2_1.0_lr_1.0E-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/size_64/n_element_2',
                              'f_XRF_data': 'simulation_XRF_data.h5',                 
                              'f_XRT_data': 'simulation_XRT_data.h5', 
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3, 
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Ca": 20, "Sc": 21},
                              'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                              'n_line_group_each_element': np.array([1,1]),
                              'sample_size_n': 64, 
                              'sample_height_n': 64,
                              'sample_size_cm': 0.01,                                    
                              'probe_energy': np.array([20.0]),                            
                              'n_epochs': 2,
                              'save_every_n_epochs': 1,
                              'minibatch_size': 64*16,
                              'b1': 1.5E-5,
                              'b2': 1.0,
                              'lr': 1.0E-3,        
                              'manual_det_coord': False,    
                              'set_det_coord_cm': None,            
                              'det_on_which_side': "positive",   
                              'det_from_sample_cm': 1.6,
                              'det_ds_spacing_cm': 0.4,
                              'manual_det_area': False,
                              'det_area_cm2': None, 
                              'det_dia_cm': 0.9,                       
                              'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                              'f_P': 'Intersecting_Length_64_64_64',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_3d_size_128 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': False,
                              'probe_intensity': 1.0E7,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': None,
                              'density_std_elements': None, 
                              'fitting_method':None,                             
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/size_128_recon/Ab_T_nEl_2_Dis_1.6_Dpts_5_b1_1.5E-5_b2_1.0_lr_1.0E-3_nG1_nRank16_nMini1',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/size_128', 
                              'f_XRF_data': 'simulation_XRF_data.h5',               
                              'f_XRT_data': 'simulation_XRT_data.h5',
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3, 
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Ca": 20, "Sc": 21},
                              'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                              'n_line_group_each_element': np.array([1,1]),
                              'sample_size_n': 128, 
                              'sample_height_n': 128,
                              'sample_size_cm': 0.01,                                    
                              'probe_energy': np.array([20.0]),                            
                              'n_epochs': 2,
                              'save_every_n_epochs': 1,
                              'minibatch_size': 128*1, #In turns of number of strips
                              'b1': 1.5E-5,
                              'b2': 1.0,
                              'lr': 1.0E-3,        
                              'manual_det_coord': False,    
                              'set_det_coord_cm': None,            
                              'det_on_which_side': "positive",   
                              'det_from_sample_cm': 1.6, 
                              'det_ds_spacing_cm': 0.4,
                              'manual_det_area': False,
                              'det_area_cm2': None, 
                              'det_dia_cm': 0.9, # The estimated diameter of the sensor                          
                              'P_folder': './data/P_array/sample_128_128_128/detSpacing_0.4_dpts_5',              
                              'f_P': 'Intersecting_Length_128_128_128',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_3d_size_256 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': False,
                              'probe_intensity': 1.0E7,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': None,
                              'density_std_elements': None,
                              'fitting_method':None,                            
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/size_256_recon/Ab_T_nEl_2_Dis_1.6_nDpts_5_b1_1.5E-5_b2_1.0_lr_1.0E-3_nRank16_nMini1',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/size_256', 
                              'f_XRF_data': 'simulation_XRF_data.h5',                  
                              'f_XRT_data': 'simulation_XRT_data.h5', 
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3, 
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Ca": 20, "Sc": 21},
                              'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                              'n_line_group_each_element': np.array([1,1]),
                              'sample_size_n': 256, 
                              'sample_height_n': 256,
                              'sample_size_cm': 0.01,                                    
                              'probe_energy': np.array([20.0]),                            
                              'n_epochs': 2,
                              'save_every_n_epochs': 1,
                              'minibatch_size': 256*1,
                              'b1': 1.5E-5,  # the regulizer coefficient of the XRT loss
                              'b2': 1.0,
                              'lr': 1.0E-3,        
                              'manual_det_coord': False,    
                              'set_det_coord_cm': None,            
                              'det_on_which_side': "positive",   
                              'det_from_sample_cm': 1.6, 
                              'det_ds_spacing_cm': 0.4,
                              'manual_det_area': False,
                              'det_area_cm2': None, 
                              'det_dia_cm': 0.9,                         
                              'P_folder': './data/P_array/sample_256_256_256/detSpacing_0.4_dpts_5',              
                              'f_P': 'Intersecting_Length_256_256_256', 
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }



params = params_3d_size_32

if __name__ == "__main__": 
    
    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        create_summary(output_folder, params)



