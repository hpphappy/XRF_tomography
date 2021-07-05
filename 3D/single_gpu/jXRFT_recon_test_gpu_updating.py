#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch as tc
import xraylib as xlib

from XRF_tomography_test_gpu_updating import reconstruct_jXRFT_tomography

import warnings
warnings.filterwarnings("ignore")


#========================================================
# Set the device
#========================================================
if tc.cuda.is_available():  
    dev = "cuda:0" 
    print("running on GPU")
else:  
    dev = "cpu"
    print("running on CPU")

device = tc.device(dev)
print("device = %s" %device)

# dev = "cpu"


fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE])

fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE])

fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

fl_line_groups = np.array(["K", "L", "M"])
group_lines = True



params_3d_5_5_5 = {'dev': dev,
                   'selfAb': True,
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
                      'n_minibatch': tc.tensor(5).to(dev) ,
                      'minibatch_size': tc.tensor(5).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 1.0E-5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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


params_3d_8_8_8_nElements_2 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample6_size_8_recon/nElements_2',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample6_size_8_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample6_size_8_data/nElements_2',
                      'f_XRF_data': 'XRF_sample6',                     
                      'f_XRT_data': 'XRT_sample6',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(8).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(25).to(dev),
                      'sample_size_n': tc.tensor(8).to(dev), 
                      'sample_height_n': tc.tensor(8).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.24,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.1,
                      'f_P': 'Intersecting_Length_8_8_8',
                     }


params_3d_16_16_16_nElements_2 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample4_size_16_recon/nElements_2',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample4_size_16_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample4_size_16_data/nElements_2',
                      'f_XRF_data': 'XRF_sample4',                     
                      'f_XRT_data': 'XRT_sample4',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(16).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(50).to(dev),
                      'sample_size_n': tc.tensor(16).to(dev), 
                      'sample_height_n': tc.tensor(16).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.24,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.1,
                      'f_P': 'Intersecting_Length_16_16_16',
                     }

params_3d_32_32_32_nElements_2 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample5_size_32_recon/nElements_2',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample5_size_32_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample5_size_32_data/nElements_2',
                      'f_XRF_data': 'XRF_sample5',                     
                      'f_XRT_data': 'XRT_sample5',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(32).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(100).to(dev),
                      'sample_size_n': tc.tensor(32).to(dev), 
                      'sample_height_n': tc.tensor(32).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.24,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.1,
                      'f_P': 'Intersecting_Length_32_32_32',
                     }

params_3d_64_64_64 = {'dev': dev,
                      'selfAb': True,
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
                      'n_epoch': tc.tensor(20).to(dev),
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
                      'theta_st': tc.tensor(0.).to(dev),
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

params_3d_64_64_64_nElements_1 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample2_recon/nElements_1',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample2_pad/nElements_1',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample2_data/nElements_1',
                      'f_XRF_data': 'XRF_sample2',                     
                      'f_XRT_data': 'XRT_sample2',
                      'this_aN_dic': {"Ca": 20},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(64).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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

params_3d_64_64_64_nElements_2 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample2_recon/nElements_2',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample2_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample2_data/nElements_2',
                      'f_XRF_data': 'XRF_sample2',                     
                      'f_XRT_data': 'XRT_sample2',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(1024).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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


params_3d_64_64_64_nElements_4 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample2_recon/nElements_4',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample2_pad/nElements_4',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample2_data/nElements_4',
                      'f_XRF_data': 'XRF_sample2',                     
                      'f_XRT_data': 'XRT_sample2',
                      'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23},    
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(64).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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

params_3d_64_64_64_nElements_8 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample2_recon/nElements_8',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample2_pad/nElements_8',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample2_data/nElements_8',
                      'f_XRF_data': 'XRF_sample2',                     
                      'f_XRT_data': 'XRT_sample2',
                      'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27},   
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(64).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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



params_3d_64_64_64_nElements_12 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample2_recon/nElements_12',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample2_pad/nElements_12',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample2_data/nElements_12',
                      'f_XRF_data': 'XRF_sample2',                     
                      'f_XRT_data': 'XRT_sample2',
                      'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27,
                                      "Ni": 28, "Cu":29, "Zn":30, "Mo":42},  
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(64).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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


params_3d_64_64_64_nElements_2 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample2_recon/nElements_2',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample2_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample2_data/nElements_2',
                      'f_XRF_data': 'XRF_sample2',                     
                      'f_XRT_data': 'XRT_sample2',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(20).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(1024).to(dev),
                      'b': 1.0E-3,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
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


params_3d_64_64_64_nElements_2_2 = {'dev': dev,
                      'selfAb': True,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': '../data/sample8_size_64_recon/limited_solid_angle/Noise/detSpacing_0.4_dpts_5/b_1.56E-5/nElements_2_selfAb_nEpochs_40_nThetas_200_singleGPU_Pbackup5_monitoring',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': '../data/sample8_size_64_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': '../data/sample8_size_64_data/nElements_2/nThetas_200_limitedSolidAngle/solidAngle_frac_0.0156/Noise',
                      'f_XRF_data': 'XRF_sample8',                     
                      'f_XRT_data': 'XRT_sample8',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(40).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(64).to(dev),
                      'b': 1.56E-5,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(200).to(dev),
                      'sample_size_n': tc.tensor(64).to(dev), 
                      'sample_height_n': tc.tensor(64).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.9,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.4,
                      'P_folder': '../data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5/backup5',            
                      'f_P': 'Intersecting_Length_64_64_64',
                     }

params_3d_64_64_64_nElements_2_3 = {'dev': dev,
                      'selfAb': False,
                      'recon_idx': 0,
                      'cont_from_check_point': False,
                      'use_saved_initial_guess': False,
                      'recon_path': 'data/sample8_size_64_recon/limited_solid_angle/Noise/detSpacing_0.4_dpts_5/b_1.56E-5/nElements_2_woSelfAb_nEpochs_40_nThetas_200',
                      'f_initial_guess': 'initialized_grid_concentration',
                      'f_recon_grid': 'grid_concentration',
                      'grid_path': './data/sample8_size_64_pad/nElements_2',
                      'f_grid': 'grid_concentration.npy',
                      'data_path': './data/sample8_size_64_data/nElements_2/nThetas_200_limitedSolidAngle/solidAngle_frac_0.0156/Noise',
                      'f_XRF_data': 'XRF_sample8',                     
                      'f_XRT_data': 'XRT_sample8',
                      'this_aN_dic': {"Ca": 20, "Sc": 21},     
                      'ini_kind': 'const',
                      'f_recon_parameters': 'recon_parameters.txt',
                      'n_epoch': tc.tensor(40).to(dev),
                      'n_minibatch': tc.tensor(1).to(dev),
                      'minibatch_size': tc.tensor(64).to(dev),
                      'b': 1.56E-5,
                      'lr': 1.0E-3,
                      'init_const': 0.5,
                      'fl_line_groups': np.array(["K", "L", "M"]),
                      'fl_K': fl_K,
                      'fl_L': fl_L,                      
                      'fl_M': fl_M,
                      'group_lines': True,   
                      'theta_st': tc.tensor(0.).to(dev),
                      'theta_end': tc.tensor(2 * np.pi).to(dev),
                      'n_theta': tc.tensor(200).to(dev),
                      'sample_size_n': tc.tensor(64).to(dev), 
                      'sample_height_n': tc.tensor(64).to(dev),
                      'sample_size_cm': tc.tensor(0.01).to(dev),
                      'probe_energy': np.array([20.0]), 
                      'probe_cts': tc.tensor(1.0E7).to(dev),
                      'det_size_cm': 0.9,
                      'det_from_sample_cm': 1.6, 
                      'det_ds_spacing_cm': 0.4,
                      'P_folder': 'data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5/backup5',  
                      'f_P': 'Intersecting_Length_64_64_64',
                     }

params = params_3d_64_64_64_nElements_2_2

if __name__ == "__main__":   
    
    reconstruct_jXRFT_tomography(**params)





