#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""

import os
import sys
import datetime
import numpy as np
import h5py
from mpi4py import MPI
import xraylib as xlib
import xraylib_np as xlib_np

import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import time
from data_generation_fns_mpi_updating_realData import rotate, MakeFLlinesDictionary_manual, intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual
from standard_calibration import calibrate_incident_probe_intensity
from misc_mpi_updating_realData import print_flush_root, print_flush_all
from forward_model_mpi_updating_realData import PPM
import warnings
warnings.filterwarnings("ignore")

    
def generate_reconstructed_FL_signal(dev, std_path, f_std, fitting_method, std_element_lines_roi, density_std_elements, selfAb,
                                     recon_path, f_recon_grid, f_reconstructed_XRF_signal, f_reconstructed_XRT_signal,
                                     data_path, f_XRT_data,
                                     photon_counts_us_ic_dataset_idx, abs_ic_dataset_idx,
                                     this_aN_dic, element_lines_roi, n_line_group_each_element, 
                                     sample_size_n, sample_height_n, sample_size_cm, probe_energy,
                                     minibatch_size,
                                     det_size_cm, det_from_sample_cm, manual_det_coord, set_det_coord_cm, det_on_which_side,
                                     manual_det_area, set_det_area_cm2,
                                     det_ds_spacing_cm, solid_angle_adjustment_factor,
                                     P_folder, f_P, fl_K, fl_L, fl_M):
    
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    
    dia_len_n = int(1.2 * (sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5) #dev
    n_voxel_batch = minibatch_size * sample_size_n #dev
    n_voxel = sample_height_n * sample_size_n**2 #dev

    #### create the file handle for experimental data; y1: channel data, y2: scalers data ####
    y2_true_handle = h5py.File(os.path.join(data_path, f_XRT_data), 'r')  
    ####----------------------------------------------------------------------------------####    
    
    #### Calculate the number of elements in the reconstructed object, list the atomic numbers ####
    n_element = len(this_aN_dic)
    aN_ls = np.array(list(this_aN_dic.values()))
    ####--------------------------------------------------------------####
    
    #### Make the lookup table of the fluorescence lines of interests ####
    fl_all_lines_dic = MakeFLlinesDictionary_manual(element_lines_roi,                           
                                                    n_line_group_each_element, probe_energy, 
                                                    sample_size_n, sample_size_cm,
                                                    fl_line_groups = np.array(["K", "L", "M"]), fl_K = fl_K, fl_L = fl_L, fl_M = fl_M) #cpu
    
    stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
    
    detected_fl_unit_concentration = tc.as_tensor(fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(dev)
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev) #dev
    
    n_line_group_each_element = tc.IntTensor(fl_all_lines_dic["n_line_group_each_element"]).to(dev)
    n_lines = fl_all_lines_dic["n_lines"] #scalar
    ####--------------------------------------------------------------####
    
    #### Calculate the MAC of probe ####
    probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, probe_energy).flatten()).to(dev)
    ####----------------------------####
    
    #### Load all object angles ####
    theta_ls = tc.from_numpy(y2_true_handle['exchange/theta'][...] * np.pi / 180).float()  #unit: rad #cpu
    n_theta = len(theta_ls)  
    ####------------------------####
    
    
    #### pick the probe photon counts before the ion chamber from the scalers data as the incoming probe photon counts
    probe_cts = calibrate_incident_probe_intensity(std_path, f_std, fitting_method, std_element_lines_roi, density_std_elements, probe_energy)

    minibatch_ls_0 = tc.arange(n_ranks).to(dev) #dev
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * minibatch_size) #scalar
     
    if manual_det_area == True:
#         fl_sig_collecting_ratio = set_det_area_cm2 / (4 * np.pi * det_from_sample_cm**2)
        fl_sig_collecting_ratio = 1.0
    
    else:
#         #### Calculate the detecting solid angle covered by the area of the spherical cap covered by the detector #### 
#         # The distance from the sample to the boundary of the detector
#         r = (det_from_sample_cm**2 + (det_size_cm/2)**2)**0.5   
#         # The height of the cap
#         h =  r - det_from_sample_cm
#         # The area of the cap area
#         fl_sig_collecting_cap_area = np.pi*((det_size_cm/2)**2 + h**2)
#         # The ratio of the detecting solid angle / full soilid angle
#         fl_sig_collecting_ratio = fl_sig_collecting_cap_area / (4*np.pi*r**2)  
        fl_sig_collecting_ratio = ((np.pi * (det_size_cm/2)**2) / det_from_sample_cm**2)/(4*np.pi)
    
    P_save_path = os.path.join(P_folder, f_P)   
    P_handle = h5py.File(P_save_path + ".h5", 'r')

    if rank == 0:
        X = np.load(os.path.join(recon_path, f_recon_grid)+'.npy').astype(np.float32)
        X = tc.from_numpy(X).float() #cpu 
        y1_hat_tot = np.zeros((n_theta, n_lines, sample_height_n * sample_size_n))
        y2_hat_tot = np.zeros((n_theta, sample_height_n * sample_size_n))

    else:
        X = None
    # bcast X from rank0 cpu to other ranks cpu and then transfer to dev
    comm.Barrier()
    X = comm.bcast(X, root=0).to(dev)         
    
    for theta_idx, theta in enumerate(theta_ls):
        stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
        timestr = str(datetime.datetime.today())     
        print_flush_root(rank, val=f"theta_idx: {theta_idx}, time: {timestr}", output_file='', **stdout_options)
            
        ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
        if selfAb == True:
            X_ap_rot = rotate(X, theta, dev) #dev
            lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
            lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev

        else:
            lac = 0.

        for m in range(n_batch):                    
            minibatch_ls = n_ranks * m + minibatch_ls_0  #dev, e.g. [5,6,7,8]
            p = minibatch_ls[rank]

            if selfAb == True:
                P_minibatch = tc.from_numpy(P_handle['P_array'][:,:, p * dia_len_n * minibatch_size * sample_size_n: (p+1) * dia_len_n * minibatch_size * sample_size_n]).to(dev)
                n_det = P_minibatch.shape[0] 

            else:
                P_minibatch = 0
                n_det = 0
      

            model = PPM(dev, selfAb, lac, X, p, n_element, n_lines, FL_line_attCS_ls,
                         detected_fl_unit_concentration, n_line_group_each_element,
                         sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                         probe_energy, probe_cts, probe_attCS_ls,
                         theta, solid_angle_adjustment_factor,
                         n_det, P_minibatch, det_size_cm, det_from_sample_cm, fl_sig_collecting_ratio)
                   
            y1_hat, y2_hat = model() #y1_hat dimension: (n_lines, minibatch_size); y2_hat dimension: (minibatch_size,)            
            y1_hat_this_batch = comm.gather(y1_hat,root=0) #dev
            y2_hat_this_batch = comm.gather(y2_hat,root=0) #dev

            if rank == 0: 
                y1_hat_this_batch = tc.cat(y1_hat_this_batch, dim=-1).detach().numpy()
                y2_hat_this_batch = tc.cat(y2_hat_this_batch, dim=-1).detach().numpy()

                sys.stdout.flush()
                y1_hat_tot[theta_idx, :, minibatch_size * p: minibatch_size * (p + n_ranks)] = y1_hat_this_batch
                y2_hat_tot[theta_idx, minibatch_size * p: minibatch_size * (p + n_ranks)] = y2_hat_this_batch             
                
            del model
            comm.Barrier()

        del lac
        tc.cuda.empty_cache()
    
    if rank == 0:
        np.save(os.path.join(recon_path, f_reconstructed_XRF_signal+".npy"), y1_hat_tot)
        np.save(os.path.join(recon_path, f_reconstructed_XRT_signal+".npy"), y2_hat_tot)
                              
    ## It's important to close the hdf5 file hadle in the end of the reconstruction.
    P_handle.close()         