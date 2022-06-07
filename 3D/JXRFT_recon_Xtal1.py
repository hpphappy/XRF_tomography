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



params_3d_44_44_20_xtal1 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds4_recon_h5test/Ab_F_nEl_4_Dis_2.0_nDpts_4_b1_1.0_b2_25000_lr_1.0E-5_test',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds4',
                              'f_XRF_data': 'xtal1_xrf-roi-plus',                 
                              'f_XRT_data': 'xtal1_scalers',
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,   
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',  
                              'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                              'element_lines_roi': np.array([['Al', 'K'], ['Si', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  # np.array([["Si, K"], ["Ca, K"]])
                              'n_line_group_each_element': np.array([1, 1, 1, 1]),
                              'sample_size_n': 44, 
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 10,
                              'save_every_n_epochs': 10,
                              'minibatch_size': 44,
                              'b1': 1.0, 
                              'b2': 25000.0,
                              'lr': 1.0E-5,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                              'det_on_which_side': "negative",
                              'det_from_sample_cm': None,
                              'det_ds_spacing_cm': None,
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,
                              'det_dia_cm': None,
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_44_44_20',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }



params_3d_44_44_20_xtal1_manual_I_probe = {
                              'f_recon_parameters': 'recon_parameters.txt', 
                              'dev': dev,
                              'use_std_calibation': False,
                              'probe_intensity': 1.0E6,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds4_recon_h5test/Ab_T_nEl_4_Dis_2.0_nDpts_4_b1_1.0_b2_25000_lr_1.0E-3_manual_Iprobe_1e6',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds4', 
                              'f_XRF_data': 'xtal1_xrf-roi-plus',                  
                              'f_XRT_data': 'xtal1_scalers',  
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3, 
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements',
                              'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                              'element_lines_roi': np.array([['Al', 'K'], ['Si', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'n_line_group_each_element': np.array([1, 1, 1, 1]),
                              'sample_size_n': 44, 
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 80,
                              'save_every_n_epochs': 10,
                              'minibatch_size': 44,
                              'b1': 1.0,  # the regulizer coefficient of the XRT loss
                              'b2': 25000.0,
                              'lr': 1.0E-3,                          
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                              'det_on_which_side': "negative", 
                              'det_from_sample_cm': None,
                              'det_ds_spacing_cm': None, 
                              'manual_det_area': True, 
                              'det_area_cm2': 1.68,
                              'det_dia_cm': None,
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_44_44_20', 
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_3d_44_44_20_xtal1_2 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': True,
                              'ini_kind': 'const', 
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted3_ds4_recon/Ab_T_nEl_4_Dis_1.0_nDpts_16_b1_0.0_lr_1E-3_ini_2',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted3_ds4',  
                              'f_XRF_data': 'xtal1_xrf-fits',                   
                              'f_XRT_data': 'xtal1_scalers', 
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3, 
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                              'element_lines_roi': np.array([['Al', 'K'], ['Si', 'K'], ['Fe', 'K'], ['Cu', 'K']]), 
                              'n_line_group_each_element': np.array([1, 1, 1, 1]),
                              'sample_size_n': 44, 
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 200,
                              'minibatch_size': 44,
                              'b1': 0.0, 
                              'b2': 25000.,        
                              'lr': 1.0E-3,                          
                              'det_dia_cm': 2.4, 
                              'det_from_sample_cm': 1.0,
                              'manual_det_coord': True,
                              'set_det_coord_cm':np.array([[0.8, -1.0, 0.8], [0.8, -1.0, 0.6], [0.6, -1.0, 0.6], [0.60, -1.0, 0.8],
                                                           [0.8, -1.0, -0.6], [0.8, -1.0, -0.8], [0.6, -1.0, -0.8], [0.6, -1.0, -0.6],
                                                           [-0.6, -1.0, -0.6], [-0.6, -1.0, -0.8], [-0.8, -1.0, -0.8], [-0.8, -1.0, -0.6],                                       
                                                           [-0.6, -1.0, 0.8], [-0.6, -1.0, 0.6], [-0.8, -1.0, 0.6], [-0.8, -1.0, 0.8]]),   
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,  
                              'det_on_which_side': "negative",
                              'det_ds_spacing_cm': 2.4/2.0,
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_1.0_detSize_2.4_manual_dpts_16',              
                              'f_P': 'Intersecting_Length_44_44_20',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_3d_44_44_20_Al_xtal1 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2
                              'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus'
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds4_recon/Ab_F_nEl_1_Dis_2.0_nDpts_4_b1_0.0_b2_25000_lr_1.0E-5/Al',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds4',
                              'f_XRF_data': 'xtal1_xrf-roi-plus',                  
                              'f_XRT_data': 'xtal1_scalers',    
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,   
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Al": 13},
                              'element_lines_roi': np.array([['Al', 'K']]),
                              'n_line_group_each_element': np.array([1]),
                              'sample_size_n': 44, 
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 40,
                              'save_every_n_epochs': 10,
                              'minibatch_size': 44,
                              'b1': 0.0, 
                              'b2': 25000.0,           
                              'lr': 1.0E-5,                          
                              'det_dia_cm': None,
                              'det_from_sample_cm': None,
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]), 
                              'det_on_which_side': "negative",                            
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,                                          
                              'det_ds_spacing_cm': None, 
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_44_44_20', 
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params_3d_44_44_20_Si_xtal1 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2
                              'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus'
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds4_recon/Ab_F_nEl_1_Dis_2.0_nDpts_4_b1_0.0_b2_25000_lr_1.0E-5/Si',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds4', 
                              'f_XRF_data': 'xtal1_xrf-roi-plus',               
                              'f_XRT_data': 'xtal1_scalers',  
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,  
                              'theta_ls_dataset': 'exchange/theta',
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Si": 14},
                              'element_lines_roi': np.array([['Si', 'K']]),  # np.array([["Si, K"], ["Ca, K"]])
                              'n_line_group_each_element': np.array([1]),
                              'sample_size_n': 44, 
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 40,
                              'save_every_n_epochs': 10,
                              'minibatch_size': 44,
                              'b1': 0.0, 
                              'b2': 25000.0,
                              'lr': 1.0E-5,                              
                              'det_dia_cm': None, 
                              'det_from_sample_cm': None,
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]), 
                              'det_on_which_side': "negative",                            
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,                                          
                              'det_ds_spacing_cm': None,
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_44_44_20',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }



params_3d_44_44_20_Fe_xtal1 = {
                              'f_recon_parameters': 'recon_parameters.txt', 
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus', 
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const', 
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds4_recon/Ab_F_nEl_1_Dis_2.0_nDpts_4_b1_0.0_b2_25000_lr_1.0E-5/Fe',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds4', 
                              'f_XRF_data': 'xtal1_xrf-roi-plus',                  
                              'f_XRT_data': 'xtal1_scalers',  
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,         
                              'theta_ls_dataset': 'exchange/theta', 
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Fe": 26},
                              'element_lines_roi': np.array([['Fe', 'K']]),
                              'n_line_group_each_element': np.array([1]),
                              'sample_size_n': 44,
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 40,
                              'save_every_n_epochs': 10,
                              'minibatch_size': 44,
                              'b1': 0.0,
                              'b2': 25000.0,
                              'lr': 1.0E-5,                             
                              'det_dia_cm': None,
                              'det_from_sample_cm': None,
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]), 
                              'det_on_which_side': "negative",                            
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,                                          
                              'det_ds_spacing_cm': None, 
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_44_44_20',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }


params_3d_44_44_20_Cu_xtal1 = {
                              'f_recon_parameters': 'recon_parameters.txt',  
                              'dev': dev,
                              'use_std_calibation': True,
                              'probe_intensity': None,
                              'std_path': './data/Xtal1/axo_std',
                              'f_std': 'axo_std.h5',
                              'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                              'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6, 
                              'fitting_method':'XRF_roi_plus',
                              'selfAb': False,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.0,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds4_recon/Ab_F_nEl_1_Dis_2.0_nDpts_4_b1_0.0_b2_25000_lr_1.0E-5/Cu',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds4', 
                              'f_XRF_data': 'xtal1_xrf-roi-plus',                   
                              'f_XRT_data': 'xtal1_scalers',   
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3,  
                              'theta_ls_dataset': 'exchange/theta',
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Cu": 29},
                              'element_lines_roi': np.array([['Cu', 'K']]),
                              'n_line_group_each_element': np.array([1]),
                              'sample_size_n': 44, 
                              'sample_height_n': 20,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                             
                              'n_epochs': 40,
                              'save_every_n_epochs': 10,
                              'minibatch_size': 44,
                              'b1': 0.0, 
                              'b2': 25000.0,
                              'lr': 1.0E-5,                              
                              'det_dia_cm': None, 
                              'det_from_sample_cm': None,
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]), 
                              'det_on_which_side': "negative",                            
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,                                          
                              'det_ds_spacing_cm': None,
                              'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                              'f_P': 'Intersecting_Length_44_44_20',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }




params_3d_88_88_40_xtal1 = {
                              'f_recon_parameters': 'recon_parameters.txt',
                              'dev': dev,
                              'selfAb': True,
                              'cont_from_check_point': False,
                              'use_saved_initial_guess': False,
                              'ini_kind': 'const',
                              'init_const': 0.5,
                              'ini_rand_amp': 0.1,
                              'recon_path': './data/Xtal1_align1_adjusted1_ds2_recon/Ab_F_nEl_4_nDpts_29_b_1E-4_lr_1E-3',
                              'f_initial_guess': 'initialized_grid_concentration',
                              'f_recon_grid': 'grid_concentration',
                              'data_path': './data/Xtal1_align1_adjusted1_ds2',
                              'f_XRF_data': 'xtal1_xrf-roi-plus',                 
                              'f_XRT_data': 'xtal1_scalers', 
                              'scaler_counts_us_ic_dataset_idx':1,
                              'scaler_counts_ds_ic_dataset_idx':2,
                              'XRT_ratio_dataset_idx':3, 
                              'theta_ls_dataset': 'exchange/theta',
                              'channel_names': 'exchange/elements', 
                              'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29},
                              'element_lines_roi': np.array([['Al', 'K'], ['Si', 'K'], ['Fe', 'K'], ['Cu', 'K']]), 
                              'n_line_group_each_element': np.array([1, 1, 1, 1]),
                              'sample_size_n': 88, 
                              'sample_height_n': 40,
                              'sample_size_cm': 0.007,                                    
                              'probe_energy': np.array([10.0]),                            
                              'n_epochs': 40,
                              'minibatch_size': 88,
                              'b': 0.0, 
                              'lr': 5.0E-4,                             
                              'det_dia_cm': 2.4,
                              'det_from_sample_cm': 3.0,
                              'manual_det_coord': True,
                              'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]), 
                              'manual_det_area': True,
                              'det_area_cm2': 1.68,       
                              'det_on_which_side': "negative",                                        
                              'det_ds_spacing_cm': 2.4/2.0, 
                              'P_folder': 'data/P_array/sample_44_44_20/Dis_2.0_detSize_2.4_detSpacing_1.2_dpts_5',              
                              'f_P': 'Intersecting_Length_44_44_20',
                              'fl_K': fl["K"], # doesn't need to change 
                              'fl_L': fl["L"], # doesn't need to change                    
                              'fl_M': fl["M"]  # doesn't need to change
                             }

params = params_3d_44_44_20_xtal1

if __name__ == "__main__": 
    
    reconstruct_jXRFT_tomography(**params)
    
    if rank == 0:
        output_folder = params["recon_path"]
        create_summary(output_folder, params)



