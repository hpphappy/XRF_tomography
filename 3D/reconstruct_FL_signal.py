#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""
import os
from FL_signal_reconstruction_fn_2 import generate_reconstructed_FL_signal
import numpy as np
from mpi4py import MPI
import xraylib as xlib
import xraylib_np as xlib_np

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()


fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
      "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
      "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
     }

params_3d_44_44_20_xtal1_roi_plus = {'dev': "cpu",
                                     'use_simulation_sample': False,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False     
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False   
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False 
                                     'selfAb': False,
                                     'recon_path':"./data/Xtal1_align1_adjusted1_ds4_recon_h5test/Ab_F_nEl_4_Dis_2.0_nDpts_4_b1_1.0_b2_25000_lr_1.0E-5",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data_test",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data_test",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':110,
                                     'data_path': './data/Xtal1_align1_adjusted1_ds4',
                                     'f_XRT_data': 'xtal1_scalers',
                                     'this_aN_dic': {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29}, 
                                     'element_lines_roi': np.array([['Al', 'K'], ['Si', 'K'], ['Fe', 'K'], ['Cu', 'K']]),
                                     'n_line_group_each_element': np.array([1, 1, 1, 1]),
                                     'sample_size_n': 44, 
                                     'sample_height_n': 20,
                                     'sample_size_cm': 0.007,                                    
                                     'probe_energy': np.array([10.0]),                                       
                                     'minibatch_size': 44,                                     
                                     'manual_det_coord': True,
                                     'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                                     'det_on_which_side': "negative",                                 
                                     'manual_det_area': True,
                                     'set_det_area_cm2': 1.68,
                                     'det_size_cm': None, # The estimated diameter of the sensor; Used only when manual_det_area is False.
                                     'det_from_sample_cm': None, # The estimated spacing between the sample and the detector; Used only when manual_det_area is False.
                                     'det_ds_spacing_cm': None, # Set to the value of det_size_cm divided by a number; Used only when manual_det_area is False.
                                     'solid_angle_adjustment_factor': 1.0, # Set to 1.0 when use_simulation_sample is False
                                     'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_manual_dpts_4',              
                                     'f_P': 'Intersecting_Length_44_44_20',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change
                                    }

params_3d_44_44_20_Al_xtal1_roi_plus = {'dev': "cpu",
                                     'use_simulation_sample': False,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False                                        
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False     
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False 
                                     'selfAb': False,
                                     'recon_path': "./data/Xtal1_align1_adjusted3_ds4_recon/Ab_F_nEl_1_nDpts_4_b_0.0_lr_1.0E-3_full_solid_angle/Al",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True  
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':110,
                                     'data_path': './data/Xtal1_align1_adjusted3_ds4',
                                     'f_XRT_data': 'xtal1_scalers',
                                     'this_aN_dic': {"Al": 13}, 
                                     'element_lines_roi': np.array([['Al', 'K']]),
                                     'n_line_group_each_element': np.array([1]),
                                     'sample_size_n': 44, 
                                     'sample_height_n': 20,
                                     'sample_size_cm': 0.007,                                    
                                     'probe_energy': np.array([10.0]),                                       
                                     'minibatch_size': 44,
                                     'manual_det_coord': True,
                                     'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                                     'det_on_which_side': "negative",
                                     'manual_det_area': True,
                                     'set_det_area_cm2': 1.68,
                                     'det_size_cm': 2.4, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 2.0, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 2.4/2.0, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_4',              
                                     'f_P': 'Intersecting_Length_44_44_20',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change
                                    }

params_3d_44_44_20_Si_xtal1_roi_plus = {'dev': "cpu",
                                     'use_simulation_sample': False,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False 
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False 
                                     'selfAb': False,
                                     'recon_path': "./data/Xtal1_align1_adjusted3_ds4_recon/Ab_F_nEl_1_nDpts_4_b_0.0_lr_1.0E-3_full_solid_angle/Si_2",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True   
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':110,
                                     'data_path': './data/Xtal1_align1_adjusted3_ds4',
                                     'f_XRT_data': 'xtal1_scalers',
                                     'this_aN_dic': {"Si": 14}, 
                                     'element_lines_roi': np.array([['Si', 'K']]),
                                     'n_line_group_each_element': np.array([1]),
                                     'sample_size_n': 44, 
                                     'sample_height_n': 20,
                                     'sample_size_cm': 0.007,                                    
                                     'probe_energy': np.array([10.0]),                                       
                                     'minibatch_size': 44,                                     
                                     'manual_det_coord': True,
                                     'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                                     'det_on_which_side': "negative",
                                     'manual_det_area': True,
                                     'set_det_area_cm2': 1.68,
                                     'det_size_cm': 2.4, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 2.0, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 2.4/2.0, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_4',              
                                     'f_P': 'Intersecting_Length_44_44_20',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change
                                    }

params_3d_44_44_20_Fe_xtal1_roi_plus = {'dev': "cpu",
                                     'use_simulation_sample': False,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False   
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False 
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False  
                                     'selfAb': False,
                                     'recon_path': "./data/Xtal1_align1_adjusted3_ds4_recon/Ab_F_nEl_1_nDpts_4_b_0.0_lr_1.0E-3_full_solid_angle/Fe",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True  
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':110,
                                     'data_path': './data/Xtal1_align1_adjusted3_ds4', #used only when generate_simulation_sample is True  
                                     'f_XRT_data': 'xtal1_scalers', #used only when generate_simulation_sample is True  
                                     'this_aN_dic': {"Fe": 26}, 
                                     'element_lines_roi': np.array([['Fe', 'K']]),
                                     'n_line_group_each_element': np.array([1]),
                                     'sample_size_n': 44, 
                                     'sample_height_n': 20,
                                     'sample_size_cm': 0.007,                                    
                                     'probe_energy': np.array([10.0]),                                       
                                     'minibatch_size': 44,                                     
                                     'manual_det_coord': True,
                                     'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                                     'det_on_which_side': "negative",
                                     'manual_det_area': True,
                                     'set_det_area_cm2': 1.68,
                                     'det_size_cm': 2.4, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 2.0, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 2.4/2.0, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_4',              
                                     'f_P': 'Intersecting_Length_44_44_20',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change
                                    }

params_3d_44_44_20_Cu_xtal1_roi_plus = {'dev': "cpu",
                                     'use_simulation_sample': False,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False 
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False 
                                     'selfAb': False,
                                     'recon_path':"./data/Xtal1_align1_adjusted3_ds4_recon/Ab_F_nEl_1_nDpts_4_b_0.0_lr_1.0E-3_full_solid_angle/Cu",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True  
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':110,
                                     'data_path': './data/Xtal1_align1_adjusted3_ds4',
                                     'f_XRT_data': 'xtal1_scalers',
                                     'this_aN_dic': {"Cu": 29}, 
                                     'element_lines_roi': np.array([['Cu', 'K']]),
                                     'n_line_group_each_element': np.array([1]),
                                     'sample_size_n': 44, 
                                     'sample_height_n': 20,
                                     'sample_size_cm': 0.007,                                    
                                     'probe_energy': np.array([10.0]),                                       
                                     'minibatch_size': 44,                                     
                                     'manual_det_coord': True,
                                     'set_det_coord_cm': np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),
                                     'det_on_which_side': "negative",
                                     'manual_det_area': True,
                                     'set_det_area_cm2': 1.68,
                                     'det_size_cm': 2.4, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 2.0, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 2.4/2.0, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': 'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_4',              
                                     'f_P': 'Intersecting_Length_44_44_20',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change
                                    }

params_3d_test_sample8_64_64_64 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when use_simulation_sample is False   
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False   
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False  
                                     'selfAb': True,
                                     'recon_path':"./data/sample_8_size_64_test_recon",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True  
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':200,
                                     'data_path': './data/sample_8_size_64_test', #used only when generate_simulation_sample is True  
                                     'f_XRT_data': 'test8_xrt', #used only when generate_simulation_sample is True  
                                     'this_aN_dic': {"Ca": 20, "Sc": 21}, 
                                     'element_lines_roi': np.array([['Ca', 'K'], ['Ca', 'L'], ['Sc', 'K'], ['Sc', 'L']]),
                                     'n_line_group_each_element': np.array([2, 2]),
                                     'sample_size_n': 64, 
                                     'sample_height_n': 64,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 64,                                    
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_64_64_64',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change                                   
                                    }

params_3d_test_sample9_64_64_64 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False  
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample isFalse 
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False 
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False 
                                     'selfAb': True,
                                     'recon_path':"./data/sample_9_size_64_recon/b1_2000_b2_1.0_lr_1E-3",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True 
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':200,
                                     'data_path': './data/sample_9_size_64_data/nElements_1', #used only when generate_simulation_sample is False 
                                     'abs_ic_dataset_idx': 3, #used only when generate_simulation_sample is False 
                                     'this_aN_dic': {"Si": 14}, 
                                     'element_lines_roi': np.array([['Si', 'K'], ['Si', 'L']]),
                                     'n_line_group_each_element': np.array([2]),
                                     'sample_size_n': 64, 
                                     'sample_height_n': 64,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 64,                                     
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_64_64_64',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change  
                                    }

params_3d_test_sample9_64_64_64 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is False 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False   
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False 
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is False   
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is False 
                                     'selfAb': True,
                                     'recon_path':"./data/sample_9_size_64_recon/b1_2000_b2_1.0_lr_1E-3",
                                     'f_recon_grid': "grid_concentration",
                                     'f_reconstructed_XRF_signal':"reprojected_XRF_data",
                                     'f_reconstructed_XRT_signal':"reprojected_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True    
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':200,
                                     'data_path': './data/sample_9_size_64_data/nElements_1', #used only when generate_simulation_sample is False 
                                     'f_XRT_data': 'test9_xrt', #used only when generate_simulation_sample is False 
                                     'this_aN_dic': {"Si": 14}, 
                                     'element_lines_roi': np.array([['Si', 'K'], ['Si', 'L']]),
                                     'n_line_group_each_element': np.array([2]),
                                     'sample_size_n': 64, 
                                     'sample_height_n': 64,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 64,                                     
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_64_64_64',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change  
                                    }

params_sample_size_32 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is True 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is True    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is True  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is True  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is True  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is True 
                                     'selfAb': True,
                                     'recon_path':"./data/size_32",
                                     'f_recon_grid': "density_n_element_2",
                                     'f_reconstructed_XRF_signal':"simulation_XRF_data_test",
                                     'f_reconstructed_XRT_signal':"simulation_XRT_data_test",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 100, #used only when generate_simulation_sample is True    
                                     'cont_from_last_theta': True,
                                     'this_theta_st_idx':50,
                                     'this_theta_end_idx':100,
                                     'data_path': None, #used only when generate_simulation_sample is False 
                                     'f_XRT_data': None, #used only when generate_simulation_sample is False 
                                     'this_aN_dic': {"Ca": 20, "Sc": 21}, 
                                     'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                                     'n_line_group_each_element': np.array([1,1]),
                                     'sample_size_n': 32, 
                                     'sample_height_n': 32,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 32,                                     
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_32_32_32/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_32_32_32',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change  
                                    }

params_sample_size_64 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is True 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is True    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is True  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is True  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is True  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is True 
                                     'selfAb': True,
                                     'recon_path':"./data/size_64/n_element_2",
                                     'f_recon_grid': "density_n_element_2",
                                     'f_reconstructed_XRF_signal':"simulation_XRF_data",
                                     'f_reconstructed_XRT_signal':"simulation_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 200, #used only when generate_simulation_sample is True 
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':200,
                                     'data_path': None, #used only when generate_simulation_sample is False 
                                     'f_XRT_data': None, #used only when generate_simulation_sample is False 
                                     'this_aN_dic': {"Ca": 20, "Sc": 21}, 
                                     'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                                     'n_line_group_each_element': np.array([1,1]),
                                     'sample_size_n': 64, 
                                     'sample_height_n': 64,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 64,                                     
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_64_64_64',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change  
                                    }

params_sample_size_128 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is True 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is True    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is True  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is True  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is True  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is True 
                                     'selfAb': True,
                                     'recon_path':"./data/size_128",
                                     'f_recon_grid': "density_n_element_2",
                                     'f_reconstructed_XRF_signal':"simulation_XRF_data",
                                     'f_reconstructed_XRT_signal':"simulation_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 400, #used only when generate_simulation_sample is True 
                                     'cont_from_last_theta': False,
                                     'this_theta_st_idx':0,
                                     'this_theta_end_idx':400,
                                     'data_path': None, #used only when generate_simulation_sample is False 
                                     'f_XRT_data': None, #used only when generate_simulation_sample is False 
                                     'this_aN_dic': {"Ca": 20, "Sc": 21}, 
                                     'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                                     'n_line_group_each_element': np.array([1,1]),
                                     'sample_size_n': 128, 
                                     'sample_height_n': 128,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 128,                                     
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_128_128_128/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_128_128_128',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change  
                                    }

params_sample_size_256 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when generate_simulation_sample is True 
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is True    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is True  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is True  
                                     'std_element_lines_roi': np.array([['Ca', 'K'], ['Fe', 'K'], ['Cu', 'K']]),  #used only when use_simulation_sample is True  
                                     'density_std_elements': np.array([1.931, 0.504, 0.284])*1.0E-6,  # unit in g/cm^2,  #used only when use_simulation_sample is True 
                                     'selfAb': True,
                                     'recon_path':"./data/size_256",
                                     'f_recon_grid': "density_n_element_2",
                                     'f_reconstructed_XRF_signal':"simulation_XRF_data",
                                     'f_reconstructed_XRT_signal':"simulation_XRT_data",
                                     'theta_st': 0, #used only when generate_simulation_sample is True
                                     'theta_end': 360, #used only when generate_simulation_sample is True
                                     'n_theta': 800, #used only when generate_simulation_sample is True
                                     'cont_from_last_theta': True,
                                     'this_theta_st_idx':609,
                                     'this_theta_end_idx':800,
                                     'data_path': None, #used only when generate_simulation_sample is False 
                                     'f_XRT_data': None, #used only when generate_simulation_sample is False 
                                     'this_aN_dic': {"Ca": 20, "Sc": 21}, 
                                     'element_lines_roi': np.array([['Ca', 'K'], ['Sc', 'K']]),
                                     'n_line_group_each_element': np.array([1,1]),
                                     'sample_size_n': 256, 
                                     'sample_height_n': 256,
                                     'sample_size_cm': 0.01,                                    
                                     'probe_energy': np.array([20.0]),                                       
                                     'minibatch_size': 256,                                     
                                     'manual_det_coord': False,
                                     'set_det_coord_cm': None,
                                     'det_on_which_side': "positive",
                                     'manual_det_area': False,
                                     'set_det_area_cm2': None,
                                     'det_size_cm': 0.9, # The estimated diameter of the sensor
                                     'det_from_sample_cm': 1.6, # The estimated spacing between the sample and the detector                           
                                     'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
                                     'solid_angle_adjustment_factor': 1.0,
                                     'P_folder': './data/P_array/sample_256_256_256/detSpacing_0.4_dpts_5',              
                                     'f_P': 'Intersecting_Length_256_256_256',  # The output file name has det_size_cm and det_ds_spacing_cm and det_from_sample_cm 
                                     'fl_K': fl["K"], # doesn't need to change 
                                     'fl_L': fl["L"], # doesn't need to change                    
                                     'fl_M': fl["M"]  # doesn't need to change  
                                    }


params = params_sample_size_256

if __name__ == "__main__": 
    
    generate_reconstructed_FL_signal(**params)

    save_path = params["recon_path"]
    with open(os.path.join(save_path, 'reprojection_parameters.txt'), "w") as recon_params:
        print(str(params).replace(",", ",\n"), file=recon_params, sep=',')
