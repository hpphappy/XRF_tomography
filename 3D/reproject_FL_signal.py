#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""
import os
from FL_signal_reprojection_fn import generate_reconstructed_FL_signal
import numpy as np
from mpi4py import MPI
import xraylib as xlib
import xraylib_np as xlib_np
from misc import create_summary

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



params_3d_test_sample8_64_64_64 = {'dev': "cpu",
                                     'use_simulation_sample': True,
                                     'simulation_probe_cts': 1.0E7, #used only when use_simulation_sample is False   
                                     'std_path': './data/Xtal1/axo_std', #used only when use_simulation_sample is False    
                                     'f_std': 'axo_std.h5', #used only when use_simulation_sample is False  
                                     'fitting_method':'XRF_roi_plus', # set to 'XRF_fits' , 'XRF_roi' or 'XRF_roi_plus',  #used only when use_simulation_sample is False  
                                     'std_element_lines_roi': None,  #used only when use_simulation_sample is False   
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



params = params_3d_test_sample8_64_64_64

if __name__ == "__main__": 
    
    generate_reconstructed_FL_signal(**params)

    
    if rank == 0:
        output_folder = params["recon_path"]
        create_summary(output_folder, params, fname="reprojection_parameters.txt")
